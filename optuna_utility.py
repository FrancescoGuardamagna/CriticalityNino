import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from BasicBifurcationUtility import *
from sklearn.metrics import mean_squared_error
import ESN_utility_basic_bifurcation as ESN_bb
from matplotlib import rc, rcParams
from ZebiakCaneUtility import *
from ESN_utility import *
from CESM_utility import *
from RealData_utility import *

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"



def objective_basic_bifurcation(trial):
    
    Nx=trial.suggest_int('Nx',8,8)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    regularization=trial.suggest_float('regularization',0.00000001,0.01)

    x=0.1
    y=0.1

    initial_conditions=[x,y]
    external_iterations=3
    amount_training_data=2000
    steps_skip=1000
    noise_amplitude=0.08
    lead_time=6

    performances_final=[]

    u_values=[-0.3,0.3]

    for i in range(external_iterations):
        print("iteration:{}".format(i))

        esn=ESN_bb.ESN(Nx=Nx,alpha=alpha,density_W=density_W,sp=sp,input_variables_number=2,input_scaling=input_scaling,regularization=regularization)
        performance_mean=[]

        for u in u_values:

            ds=basic_bifurcation(u,1,0.1,noise_amplitude)
            data=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)

            training_data=data[steps_skip:,:]
            training_labels=training_data[lead_time:,:].copy()
            training_data=training_data[:-lead_time,:].copy()
            
            esn_representation_train=esn.esn_transformation(training_data)

            if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                continue
            
            esn.train(esn_representation_train,training_labels)

            data_validation=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)

            data_validation=np.array(data_validation)
            data_validation=data_validation[steps_skip:,:]
            y_validation=data_validation[lead_time:,:].copy()
            X_validation=data_validation[:-lead_time,:].copy()

            X_validation_esn_representation=esn.esn_transformation(X_validation)

            if(np.isnan(X_validation_esn_representation).any() or np.isinf(X_validation_esn_representation).any()):
                continue

            predictions=esn.predict(X_validation_esn_representation)
            result=mean_squared_error(predictions,y_validation)
            
            performance_mean.append(result)
        
        del esn
        
        if(len(performance_mean)==len(u_values)):

            performances_final.append(np.mean(np.array(performance_mean)))

    
    if(len(performances_final)!=external_iterations):
        print("unstable results for this set of parameters")
        return 100
    else:
        return np.mean(np.array(performances_final))
        
        

def objective_Zebiak_Cane(trial):

    Nx=trial.suggest_int('Nx',5,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    bias_scaling=trial.suggest_float('bias_scaling',-2.0,2.0)
    regularization=trial.suggest_float('regularization',0.00001,0.1)
    
    transient_skip=960
    external_iterations=3
    indexes_predictors=[1,2,3,4]
    amount_training_data=1200

    performances_final=[]

    mu_values=[2.5,2.6,2.7,2.8,3.1]
    noise=1

    for i in range(external_iterations):

        print("iteration:{}".format(i))

        esn=ESN(Nx,alpha,density_W,sp,len(indexes_predictors),input_scaling,bias_scaling,regularization)
        performance_mean=[]

        for mu_tmp in mu_values:

            zc=ZebiakCane(mu_tmp,noise)

            data=zc.load_Zebiak_Cane_data("run"+str(i))
            data=np.array(data)
           
            training_data=data[transient_skip:amount_training_data,indexes_predictors]
            
            training_labels=training_data[6:,:].copy()
            training_data=training_data[:-6,:].copy()
            
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
            y_validation=data_validation[6:,:]
            X_validation=data_validation[:-6,:]

            X_validation_esn_representation=esn.esn_transformation(X_validation)

            if(np.isnan(X_validation_esn_representation).any() or np.isinf(X_validation_esn_representation).any()):
                continue
            predictions=esn.predict(X_validation_esn_representation)
            result=esn.evaluate(predictions,y_validation,multioutput=True)
            
            performance_mean.append(result)
        
        del esn
        
        if(len(performance_mean)==len(mu_values)):

            performances_final.append(np.mean(np.array(performance_mean)))

    
    if(len(performances_final)!=external_iterations):
        print("unstable results for this set of parameters")
        return 100
    else:
        return np.mean(np.array(performances_final))  

def objective_Jin_Timmerman(trial):
    
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
    amount_training_data=1800
    noise_constant=False
    noise_amplitude=0.0095

    performances_final=[]

    delta_values=[0.194,0.214]

    for i in range(external_iterations):
        print("iteration:{}".format(i))

        esn=ESN(Nx=Nx,alpha=alpha,density_W=density_W,sp=sp,input_variables_number=len(indexes_predictors),
        input_scaling=input_scaling,reservoir_scaling=reservoir_scaling,bias_scaling=bias_scaling,
        regularization=regularization)
        performance_mean=[]

        for delta_tmp in delta_values:

            ds=DS(delta=delta_tmp,noise_amplitude=noise_amplitude,noise_constant=noise_constant)

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

            data_validation=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)

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
            result=esn.evaluate(predictions,y_validation,multioutput=True)
            
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

    Nx=trial.suggest_int('Nx',20,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    bias_scaling=trial.suggest_float('bias_scaling',-2.0,2.0)
    regularization=trial.suggest_float('regularization',0.00000001,0.01)
    
    variables=['TF','ZW','STE']
    variables_number=3

    cesm=CESM()
    last_year=1300

    data=cesm.data_no_seasonal_moving.copy()
    data['TF']=data['TF'].copy()/10
    data.drop(data.index[(data["year"]>(last_year-20))],axis=0,inplace=True)
    last_year=last_year-20

    data=np.array(data[variables])
    data=np.array(data)
    external_iterations=3

    lead_times=[3,6,9]

    performances_leads=[]
    weights=True

    if(weights):

        amount_weights=trial.suggest_float('training_weights',0.5,1)
    
    else:

        amount_weights=1

    for lead_time in lead_times:

        print("analyzing lead:{}".format(lead_time))

        labels_total=data[lead_time:,:]

        training_mean=np.mean(labels_total[:,-1])
        training_std=np.std(labels_total[:,-1])

        data_tmp=data[:-lead_time,:]
        training_weights_total=General_utility.return_classes_weights(labels_total,training_mean,training_std,amount_weights)

        performances_final=[]
    
        for iteration in range(external_iterations):

            esn=ESN(Nx,alpha,density_W,sp,variables_number,input_scaling,bias_scaling,regularization)
            performance_mean=[]

            number_split=3      
            tscv=TimeSeriesSplit(number_split)  

            for train_indexes,validation_indexes in tscv.split(data_tmp):

                scaler=StandardScaler()
                training_data=data_tmp[train_indexes,:]
                training_labels=labels_total[train_indexes,:]

                if(weights):
                    training_weights=training_weights_total[train_indexes]
                else:
                    training_weights=None

                scaler.fit(training_data)
                training_data=scaler.transform(training_data)
                esn_representation_train=esn.esn_transformation(training_data)

                if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                    continue
                
                esn.train(esn_representation_train,training_labels,weights=training_weights)

                validation_data=data_tmp[validation_indexes,:]
                validation_labels=labels_total[validation_indexes,:] 
                validation_data=scaler.transform(validation_data)
                esn_representation_validation=esn.esn_transformation(validation_data)

                if(np.isnan(esn_representation_validation).any() or np.isinf(esn_representation_validation).any()):
                    continue

                predictions=esn.predict(esn_representation_validation)
                result=esn.evaluate(predictions,validation_labels,multioutput=True)
                performance_mean.append(result)
            
            del esn

            if(len(performance_mean)==number_split):

                performances_final.append(np.mean(np.array(performance_mean)))

        if(len(performances_final)!=external_iterations):

            print("unstable results for this set of parameters")
            return 10000

        else:

            performances_leads.append(np.mean(np.array(performances_final)))
        
    return np.mean(np.array(performances_leads))


def objectiveRealData(trial):

    Nx=trial.suggest_int('Nx',70,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,1.0)
    bias_scaling=trial.suggest_float('bias_scaling',-10.0,10.0)
    regularization=trial.suggest_float('regularization',0.00000001,0.01)

    variables=['TF','ZW','STE']
    variables_number=3

    real=RealData()

    last_year=2020

    data=real.data
    data.drop(data.index[(data["year"]>(last_year-20))],axis=0,inplace=True)
    last_year=last_year-20

    data=np.array(data[variables])

    data=np.array(data)
    external_iterations=5

    lead_times=[9]

    performances_leads=[]

    weights=True

    if(weights):

        amount_weights=trial.suggest_float('training_weights',0.5,1)
    
    else:

        amount_weights=1
    

    for lead_time in lead_times:

        print("analyzing lead:{}".format(lead_time))

        labels_total=data[lead_time:,:]

        training_mean=np.mean(labels_total[:,-1])
        training_std=np.std(labels_total[:,-1])

        data_tmp=data[:-lead_time,:]

        if(weights):
            training_weights_total=General_utility.return_classes_weights(labels_total,training_mean,training_std,amount_weights)

        performances_final=[]
    
        for iteration in range(external_iterations):

            esn=ESN(Nx,alpha,density_W,sp,variables_number,input_scaling,bias_scaling,regularization,amount_weights)
            performance_mean=[]

            number_split=5

            tscv=TimeSeriesSplit(number_split)  

            for train_indexes,validation_indexes in tscv.split(data_tmp):

                scaler=StandardScaler()
                training_data=data_tmp[train_indexes,:]

                training_labels=labels_total[train_indexes,:]

                if(weights):
                    training_weights=training_weights_total[train_indexes]
                else:
                    training_weights=None

                scaler.fit(training_data)
                training_data=scaler.transform(training_data)
                
                esn_representation_train=esn.esn_transformation(training_data)

                if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                    continue
                
                esn.train(esn_representation_train,training_labels,weights=training_weights)


                validation_data=data_tmp[validation_indexes,:]
                
                validation_labels=labels_total[validation_indexes,:]
                validation_data=scaler.transform(validation_data)

                esn_representation_validation=esn.esn_transformation(validation_data)

                if(np.isnan(esn_representation_validation).any() or np.isinf(esn_representation_validation).any()):
                    continue

                predictions=esn.predict(esn_representation_validation)
                result=esn.evaluate(predictions,validation_labels,pearsons=False,multioutput=True)
                performance_mean.append(result)
            
            del esn

            if(len(performance_mean)==number_split):

                performances_final.append(np.mean(np.array(performance_mean)))

        if(len(performances_final)!=external_iterations):

            print("unstable results for this set of parameters")
            return 10000

        else:

            performances_leads.append(np.mean(np.array(performances_final)))
        
    return np.mean(np.array(performances_leads))



def optuna_optimization(objective_function,direction,n_trials):

    study=optuna.create_study(direction=direction)
    if(objective_function=='BasicBifurcation'):
        study.optimize(objective_basic_bifurcation,n_trials=n_trials)
    if(objective_function=='ZebiakCane'):
        study.optimize(objective_Zebiak_Cane,n_trials=n_trials)
    if(objective_function=='JinTimmerman'):
        study.optimize(objective_Jin_Timmerman,n_trials=n_trials)
    if(objective_function=='CESM'):
        study.optimize(objectiveCesmData,n_trials=n_trials)
    if(objective_function=='Real'):
        study.optimize(objectiveRealData,n_trials=n_trials)
    
    best_trials=study.best_trials
    return best_trials

