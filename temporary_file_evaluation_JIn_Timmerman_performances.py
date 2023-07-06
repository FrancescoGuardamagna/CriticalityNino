from optuna_utility import optuna_optimization
from CESM_utility import CESM
from RealData_utility import *
import General_utility
from JinTimmermanUtility import *
from Estimating_RMS_Zebiak_Cane import *
from wrapper_best_parameters import * 
import netCDF4 as nc
from sklearn.metrics import mean_absolute_percentage_error
import random
from sklearn.preprocessing import StandardScaler
import math

def evaluation_JIn():
    wrapper=wrapper_best_parameters()
    best_params=wrapper.best_parameters_Jin_xz


    x=-2.39869 
    y=-0.892475
    z=1.67194 

    initial_conditions=[x,y,z]
    external_iterations=100
    indexes_predictors=[0,2]
    steps_skip=0
    amount_training_data=1800
    noise_constant=False
    noise_amplitude=0.0095

    performances_final_x=[]
    performances_final_y=[]
    performances_final_z=[]

    delta_values=[0.214]

    for i in range(external_iterations):
        print("iteration:{}".format(i))

        esn=ESN(**best_params,input_variables_number=len(indexes_predictors))
        performance_mean_x=[]
        performance_mean_y=[]
        performance_mean_z=[]

        for delta_tmp in delta_values:

            ds=DS(a=8,delta=delta_tmp,noise_amplitude=noise_amplitude,noise_constant=noise_constant)

            data=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)
            data=data[steps_skip:amount_training_data,:]
            data_monthly=[]

            for i in range(0,data.shape[0],3):
                data_monthly.append(np.mean(data[i:i+3,:],axis=0))

            data_monthly=np.array(data_monthly)
            training_data=data_monthly[:,indexes_predictors]
            training_labels=training_data[6:,:].copy()
            training_data=training_data[:-6,:].copy()
            
            esn_representation_train=esn.esn_transformation(training_data)

            if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                continue
            
            esn.train(esn_representation_train,training_labels)

            data_validation=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)
            data_validation=data_validation[steps_skip:amount_training_data,:]

            data_monthly_validation=[]

            for i in range(0,data_validation.shape[0],3):
                data_monthly_validation.append(np.mean(data_validation[i:i+3,:],axis=0))

            data_monthly_validation=np.array(data_monthly_validation)
            data_monthly_validation=data_monthly_validation[:,indexes_predictors]
            y_validation=data_monthly_validation[6:,:]
            X_validation=data_monthly_validation[:-6,:]

            X_validation_esn_representation=esn.esn_transformation(X_validation)

            if(np.isnan(X_validation_esn_representation).any() or np.isinf(X_validation_esn_representation).any()):
                continue
            predictions=esn.predict(X_validation_esn_representation)
            #print(predictions.shape)

            loss_for_variable=[]
            
            for variable_index in range(predictions.shape[1]):

                loss=mean_squared_error(y_validation[:,variable_index],predictions[:,variable_index])
                root_loss=math.sqrt(loss)
                #print(root_loss)
                std_target=np.std(y_validation[:,variable_index])
                #print(std_target)
                loss_for_variable.append(root_loss/std_target)

            result_x=loss_for_variable[0]
            #result_y=loss_for_variable[1]
            result_z=loss_for_variable[1]
            
            performance_mean_x.append(result_x)
            #performance_mean_y.append(result_y)
            performance_mean_z.append(result_z)

            #plt.plot(y_validation[:,0])
            #plt.plot(predictions[:,0])
            #plt.show()
        
        del esn
        
        if(len(performance_mean_x)==len(delta_values)):

            performances_final_x.append(np.mean(np.array(performance_mean_x)))
            #performances_final_y.append(np.mean(np.array(performance_mean_y)))
            performances_final_z.append(np.mean(np.array(performance_mean_z)))


    print("x:{}".format(np.mean(np.array(performances_final_x))))
    #print("y:{}".format(np.mean(np.array(performances_final_y))))
    print("z:{}".format(np.mean(np.array(performances_final_z))))



def evaluate_Zebiak_Cane():

    noise=1
    wrapper=wrapper_best_parameters()
    best_params=wrapper.bestParmasDictionaryZC40[noise]
    
    transient_skip=960
    external_iterations=100
    indexes_predictors=[1,2,3,4]
    amount_training_data=2400

    performances_final_TW=[]
    performances_final_TE=[]
    performances_final_hW=[]
    performances_final_hE=[]
    

    mu_values=[3.1]

    for i in range(external_iterations):

        print("iteration:{}".format(i))

        esn=ESN(**best_params,input_variables_number=len(indexes_predictors))
        performance_mean_TE=[]
        performance_mean_TW=[]
        performance_mean_hE=[]
        performance_mean_hw=[]

        for mu_tmp in mu_values:

            scaler=StandardScaler()

            zc=ZebiakCane(mu_tmp,noise,0.175)

            index_run=random.choice([i for i in range(zc.number_of_runs())])

            data=zc.load_Zebiak_Cane_data("run"+str(index_run))
            data=np.array(data)

            

            training_data=data[transient_skip:amount_training_data,indexes_predictors].copy()
            #training_data=training_data-np.mean(training_data,axis=0)
            training_labels=training_data[12:,:].copy()
            training_data=training_data[:-12,:].copy()
            
            esn_representation_train=esn.esn_transformation(training_data)

            if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                continue
            
            esn.train(esn_representation_train,training_labels)

            list_indexes_validation=[i for i in range(zc.number_of_runs())]
            list_indexes_validation.remove(index_run)
            index_run_validation=random.choice(list_indexes_validation)
            data_validation=zc.load_Zebiak_Cane_data("run"+str(index_run_validation))
            data_validation=np.array(data_validation)
            data_validation=data_validation[transient_skip:amount_training_data,indexes_predictors].copy()
            #data_validation=data_validation-np.mean(data_validation,axis=0)

            y_validation=data_validation[12:,:].copy()
            X_validation=data_validation[:-12,:].copy()

            X_validation_esn_representation=esn.esn_transformation(X_validation)

            if(np.isnan(X_validation_esn_representation).any() or np.isinf(X_validation_esn_representation).any()):
                continue
            predictions=esn.predict(X_validation_esn_representation)

            loss_for_variable=[]
            
            for variable_index in range(predictions.shape[1]):

                loss=mean_squared_error(y_validation[:,variable_index],predictions[:,variable_index])
                root_loss=math.sqrt(loss)
                std_target=np.std(y_validation[:,variable_index])
                loss_for_variable.append(root_loss/std_target)
            
            result_TW=loss_for_variable[0]
            result_TE=loss_for_variable[1]
            result_hW=loss_for_variable[2]
            result_hE=loss_for_variable[3]
            
            performance_mean_TW.append(result_TW)
            performance_mean_TE.append(result_TE)
            performance_mean_hw.append(result_hW)
            performance_mean_hE.append(result_hE)
        
        del esn
        
        if(len(performance_mean_TW)==len(mu_values)):

            performances_final_TW.append(np.mean(np.array(performance_mean_TW)))
            performances_final_TE.append(np.mean(np.array(performance_mean_TE)))
            performances_final_hW.append(np.mean(np.array(performance_mean_hw)))
            performances_final_hE.append(np.mean(np.array(performance_mean_hE)))

    print("TW:{}".format(np.mean(np.array(performances_final_TW))))
    print("TE:{}".format(np.mean(np.array(performances_final_TE))))
    print("hW:{}".format(np.mean(np.array(performances_final_hW))))
    print("hE:{}".format(np.mean(np.array(performances_final_hE))))


evaluate_Zebiak_Cane()