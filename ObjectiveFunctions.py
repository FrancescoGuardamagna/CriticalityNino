import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from ESN_utility import ESN
from ZebiakCaneUtility import ZebiakCane
from JinTimmermanUtility import DS
from CESM_utility import CESM


def objective_Zebiak_Cane(trial):

    Nx=trial.suggest_int('Nx',5,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    bias_scaling=trial.suggest_float('bias_scaling',-2.0,2.0)
    reservoir_scaling=trial.suggest_float('reservoir_scaling',0.1,2.0)
    regularization=trial.suggest_float('regularization',0.00001,0.1)
    
    transient_skip=960
    external_iterations=3
    indexes_predictors=[1,2,3,4]
    amount_training_data=1440

    performances_final=[]

    mu_values=[2.5,2.6,2.7,2.8,3.1]
    noise=1

    for i in range(external_iterations):

        print("iteration:{}".format(i))

        esn=ESN(Nx,alpha,density_W,sp,len(indexes_predictors),input_scaling,reservoir_scaling,bias_scaling,regularization)
        performance_mean=[]

        for mu_tmp in mu_values:


            zc=ZebiakCane(mu_tmp,noise,0.175)

            data=zc.load_Zebiak_Cane_data("run"+str(i))
            data=np.array(data)
            training_data=data[transient_skip:amount_training_data,indexes_predictors]
            training_labels=training_data[12:,:].copy()
            training_data=training_data[:-12,:].copy()
            
            esn_representation_train=esn.esn_transformation(training_data)

            if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                continue
            
            esn.train(esn_representation_train,training_labels)

            if (mu_tmp==3.1 and i==2):
                data_validation=zc.load_Zebiak_Cane_data("run0")
            else:
                data_validation=zc.load_Zebiak_Cane_data("run"+str(i+1))
            data_validation=np.array(data_validation)
            data_validation=data_validation[:,indexes_predictors]
            y_validation=data_validation[12:,:]
            X_validation=data_validation[:-12,:]

            X_validation_esn_representation=esn.esn_transformation(X_validation)

            if(np.isnan(X_validation_esn_representation).any() or np.isinf(X_validation_esn_representation).any()):
                continue
            predictions=esn.predict(X_validation_esn_representation)
            result=esn.evaluate(predictions,y_validation)
            
            performance_mean.append(result)
        
        del esn
        
        if(len(performance_mean)==len(mu_values)):

            performances_final.append(np.mean(np.array(performance_mean)))

    
    if(len(performances_final)!=external_iterations):
        print("unstable results for this set of parameters")
        return 100
    else:
        return np.mean(np.array(performances_final))

def objective(trial):
    
    Nx=trial.suggest_int('Nx',5,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    bias_scaling=trial.suggest_float('bias_scaling',-2.0,2.0)
    reservoir_scaling=trial.suggest_float('reservoir_scaling',0.1,2.0)
    regularization=trial.suggest_float('regularization',0.00000001,0.01)
    

    x=-2.39869 
    y=-0.892475
    z=1.67194 

    initial_conditions=[x,y,z]
    external_iterations=3
    indexes_predictors=[0,1,2]
    amount_training_data=4000
    noise_constant=False
    noise_amplitude=0.0095

    performances_final=[]

    delta_values=[0.18,0.194,0.211,0.214]

    for i in range(external_iterations):
        print("iteration:{}".format(i))

        esn=ESN(Nx,alpha,density_W,sp,len(indexes_predictors),input_scaling,reservoir_scaling,bias_scaling,regularization)
        performance_mean=[]

        for delta_tmp in delta_values:

            ds=DS(a=8,delta=delta_tmp,noise_amplitude=noise_amplitude,noise_constant=noise_constant)

            data=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)
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

            data_validation=ds.integrate_euler_maruyama(5500,initial_conditions=initial_conditions)

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
            result=esn.evaluate(predictions,y_validation)
            
            performance_mean.append(result)
        
        del esn
        
        if(len(performance_mean)==len(delta_values)):

            performances_final.append(np.mean(np.array(performance_mean)))

    
    if(len(performances_final)!=external_iterations):
        print("unstable results for this set of parameters")
        return 100
    else:
        return np.mean(np.array(performances_final))


def objectiveCesmData(trial):

    Nx=trial.suggest_int('Nx',5,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    bias_scaling=trial.suggest_float('bias_scaling',-2.0,2.0)
    reservoir_scaling=trial.suggest_float('reservoir_scaling',0.1,2.0)
    regularization=trial.suggest_float('regularization',0.00000001,0.01)

    cesm=CESM('low')

    dataset=cesm.data_no_seasoal
    data=dataset[['TW','TE','STW','STE']]
    data=np.array(data)
    performances_final=[]
    external_iterations=3
    
    for iteration in range(external_iterations):

        print("iteration:{}".format(iteration))

        esn=ESN(Nx,alpha,density_W,sp,4,input_scaling,reservoir_scaling,bias_scaling,regularization)
        performance_mean=[]        

        for i in range((int(data.shape[0]/240))-1):
            training_data=data[i*240:(i+1)*240,:]
            validation_data=data[(i+1)*240:(i+2)*240,:]

            training_labels=training_data[6:,:].copy()
            training_data=training_data[:-6,:].copy()
            
            esn_representation_train=esn.esn_transformation(training_data)

            if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                continue
            
            esn.train(esn_representation_train,training_labels)\

            validation_labels=validation_data[6:,:].copy()
            validation_data=validation_data[:-6,:].copy()

            esn_representation_validation=esn.esn_transformation(validation_data)

            if(np.isnan(esn_representation_validation).any() or np.isinf(esn_representation_validation).any()):
                continue

            predictions=esn.predict(esn_representation_validation)
            result=esn.evaluate(predictions,validation_labels)
            

            performance_mean.append(result)
        
        del esn

        if(len(performance_mean)==((int(data.shape[0]/240))-1)):

            performances_final.append(np.mean(np.array(performance_mean)))

    if(len(performances_final)!=external_iterations):

        print("unstable results for this set of parameters")
        return 100

    else:
        
        return np.mean(np.array(performances_final)) 