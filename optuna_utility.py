import numpy as np
import pandas as pd
import optuna
from JinTimmermanUtility import DS
from ZebiakCaneUtility import ZebiakCane
from ESN_utility import ESN
from CESM_utility import CESM
from sklearn.model_selection import TimeSeriesSplit
from RealData_utility import *
from sklearn.preprocessing import StandardScaler
import General_utility
from exploring_Wout import *
from sklearn.metrics import mean_squared_error

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
    indexes_predictors=[0,2]
    amount_training_data=1800
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

    Nx=trial.suggest_int('Nx',20,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    bias_scaling=trial.suggest_float('bias_scaling',-2.0,2.0)
    regularization=trial.suggest_float('regularization',0.00000001,0.01)
    

    cycle=False

    resolution="low"
    
    if(cycle):
        variables=['cos','sin','TF','ZW','STE']
        variables_number=5
    else:
        variables=['TF','ZW','STE']
        variables_number=3

    cesm=CESM(resolution)
    if(resolution=='low'):

        last_year=1300

    if(resolution=='high'):

        last_year=300

    data=cesm.data_no_seasonal_moving.copy()
    data['TF']=data['TF'].copy()/10
    data.drop(data.index[(data["year"]>(last_year-20))],axis=0,inplace=True)
    last_year=last_year-20

    data=np.array(data[variables])

    data=np.array(data)
    external_iterations=3

    lead_times=[3,6,9]

    performances_leads=[]

    weights=False
    weights_validation=False

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
        validation_weights_total=General_utility.return_classes_weights(labels_total,training_mean,training_std,1)

        performances_final=[]
    
        for iteration in range(external_iterations):

            esn=ESN(Nx,alpha,density_W,sp,variables_number,input_scaling,bias_scaling,regularization)
            performance_mean=[]

            number_split=3      

            tscv=TimeSeriesSplit(number_split)  

            for train_indexes,validation_indexes in tscv.split(data_tmp):

                scaler=StandardScaler()
                training_data=data_tmp[train_indexes,:]

                if(cycle):

                    training_labels=labels_total[train_indexes,2:]

                else:

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

                if(cycle):
                    validation_labels=labels_total[validation_indexes,2:]
                else:
                    validation_labels=labels_total[validation_indexes,:]

                if(weights_validation):
                    validation_weights=validation_weights_total[validation_indexes]
                else:
                    validation_weights=None
                    
                validation_data=scaler.transform(validation_data)

                esn_representation_validation=esn.esn_transformation(validation_data)

                if(np.isnan(esn_representation_validation).any() or np.isinf(esn_representation_validation).any()):
                    continue



                predictions=esn.predict(esn_representation_validation)


                result=esn.evaluate(predictions,validation_labels,weights=validation_weights,multioutput=True)
                

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
    

    cycle=False
    
    if(cycle):
        variables=['cos','sin','TF','ZW','STE']
        variables_number=5
    else:
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

                if(cycle):

                    training_labels=labels_total[train_indexes,2:]

                else:

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

                if(cycle):
                    validation_labels=labels_total[validation_indexes,2:]
                else:
                    validation_labels=labels_total[validation_indexes,:]

                if(weights):
                    validation_weights=training_weights_total[validation_indexes]
                else:
                    validation_weights=None
                    
                validation_data=scaler.transform(validation_data)

                esn_representation_validation=esn.esn_transformation(validation_data)

                if(np.isnan(esn_representation_validation).any() or np.isinf(esn_representation_validation).any()):
                    continue



                predictions=esn.predict(esn_representation_validation)


                result=esn.evaluate(predictions,validation_labels,pearsons=False,multioutput=True,weights=None)
                

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


def objectiveCesmDataDifferentPeriods(trial):

    Nx=trial.suggest_int('Nx',20,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    bias_scaling=trial.suggest_float('bias_scaling',-2.0,2.0)
    reservoir_scaling=trial.suggest_float('reservoir_scaling',0.1,2.0)
    regularization=trial.suggest_float('regularization',0.00000001,0.01)

    resolution='low'

    cesm=CESM(resolution)
    
    full=False

    dataset=cesm.data_no_seasonal_moving.copy()
    dataset['TF']=dataset['TF'].copy()/10

    if(resolution=='low'):

        first_year=1200
        last_year=1300

    if(resolution=='high'):

        first_year=200
        last_year=300

    dataset.drop(dataset.index[(dataset["year"]>(last_year-20))],axis=0,inplace=True)

    last_year=last_year-20

    train_period="1200-1240"

    interval=train_period.split('-')
    start_year=int(interval[0])
    end_year=int(interval[1])

    dataset_train=dataset.loc[(dataset['year'].astype('int')>=start_year) & (dataset['year'].astype('int')<=end_year)]

    data_train=dataset_train[['TF','ZW','STE']]
    input_variables=3

    data_train=np.array(data_train)

    lead_times=[3]
    external_iterations=3
    performances_per_lead_time=[]
    
    for lead_time in lead_times:

        training_labels=data_train[lead_time:,:].copy()
        training_data=data_train[:-lead_time,:].copy()

        if(start_year>first_year and end_year<last_year):

            dataset_1=dataset.loc[dataset['year'].astype('int')<start_year]
            dataset_2=dataset.loc[dataset['year'].astype('int')>end_year]
            data_1=dataset_1[['TF','ZW','STE']]
            data_1=np.array(data_1)
            labels_1=data_1[lead_time:,:]
            data_1=data_1[:-lead_time,:]
            data_2=dataset_2[['TF','ZW','STE']]
            data_2=np.array(data_2)
            labels_2=data_2[lead_time:,:]
            data_2=data_2[:-lead_time,:]
            validation_data=np.concatenate((data_1,np.zeros(shape=(1,data_1.shape[1])),data_2),axis=0)
            validation_labels=np.concatenate((labels_1,labels_2),axis=0)   

        if(start_year==first_year):

            dataset_validation=dataset.loc[dataset['year'].astype('int')>(end_year)].copy()
            validation_data=dataset_validation[['TF','ZW','STE']]
            validation_data=np.array(validation_data)
            validation_labels=validation_data[lead_time:,:]
            validation_data=validation_data[:-lead_time,:] 

        if(end_year==last_year):
            
            validation_dataset=dataset.loc[dataset['year'].astype('int')<start_year]
            validation_data=validation_dataset[['TF','ZW','STE']]
            validation_data=np.array(validation_data)
            validation_labels=validation_data[lead_time:,:]
            validation_data=validation_data[:-lead_time,:]
            

        print("analyzing lead time:{}".format(lead_time))

        performances_final=[]

        for _ in range(external_iterations):

            esn=ESN(Nx,alpha,density_W,sp,input_variables,input_scaling,reservoir_scaling,bias_scaling,regularization)
            performance_mean=[]

            number_split=2

            esn_representation_validation=esn.esn_transformation(validation_data)
            validation_chunks=np.array_split(esn_representation_validation,2,axis=0)
            validation_labels_chunks=np.array_split(validation_labels,2,axis=0)

            for validation_entries,validation_entries_labels in zip(validation_chunks,validation_labels_chunks):
                
                esn_representation_train=esn.esn_transformation(training_data)

                if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                    continue
                
                esn.train(esn_representation_train,training_labels)

                if(np.isnan(esn_representation_validation).any() or np.isinf(esn_representation_validation).any()):
                    continue

                predictions=esn.predict(validation_entries)
                result=esn.evaluate(predictions,validation_entries_labels)
                

                performance_mean.append(result)
            
            del esn

            if(len(performance_mean)==number_split):

                performances_final.append(np.mean(np.array(performance_mean)))

        if(len(performances_final)!=external_iterations):

            continue

        else:
            
            performances_per_lead_time.append(np.mean(np.array(performances_final))) 
    
    if(len(performances_per_lead_time)!=len(lead_times)):

        print("unstable results for this set of parameters")
        return 100
    
    else:

        return np.mean(np.array(performances_per_lead_time))


def objective_basic_bifurcation(trial):
    
    Nx=trial.suggest_int('Nx',5,1000)
    alpha=trial.suggest_float('alpha',0.1,1.0)
    density_W=trial.suggest_float('density_W',0.1,1.0)
    sp=trial.suggest_float('sp',0.1,1.0)
    input_scaling=trial.suggest_float('input_scaling',0.1,2.0)
    bias_scaling=trial.suggest_float('bias_scaling',-2.0,2.0)
    reservoir_scaling=trial.suggest_float('reservoir_scaling',0.1,2.0)
    regularization=trial.suggest_float('regularization',0.00000001,0.01)

    

    x=0.1
    y=0.1

    initial_conditions=[x,y]
    external_iterations=3
    indexes_predictors=[0,1]
    amount_training_data=4000
    steps_skip=2000
    noise_amplitude=0.08

    performances_final=[]

    u_values=[-0.08,0.1]

    for i in range(external_iterations):
        print("iteration:{}".format(i))

        esn=ESN(Nx,alpha,density_W,sp,len(indexes_predictors),input_scaling,bias_scaling,regularization)
        performance_mean=[]

        for u in u_values:

            ds=basic_bifurcation(u,1,0.1,noise_amplitude)

            data=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)

            training_data=data[steps_skip:,indexes_predictors]
            training_labels=training_data[6:,:].copy()
            training_data=training_data[:-6,:].copy()
            
            esn_representation_train=esn.esn_transformation(training_data)

            if(np.isnan(esn_representation_train).any() or np.isinf(esn_representation_train).any()):
                continue
            
            esn.train(esn_representation_train,training_labels)

            data_validation=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)

            data_validation=np.array(data_validation)
            data_validation=data_validation[steps_skip:,indexes_predictors]
            y_validation=data_validation[6:,:].copy()
            X_validation=data_validation[:-6,:].copy()

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




def optuna_optimization(objective_function,direction,n_trials):

    study=optuna.create_study(direction=direction)
    if(objective_function=='CESM'):
        study.optimize(objectiveCesmData, n_trials=n_trials)
    if(objective_function=='ZebiakCane'):
        study.optimize(objective_Zebiak_Cane, n_trials=n_trials)
    if(objective_function=='Jin'):
        study.optimize(objective, n_trials=n_trials)
    if(objective_function=='Real'):
        study.optimize(objectiveRealData,n_trials=n_trials)
    if(objective_function=='CESM_different_periods'):
        study.optimize(objectiveCesmDataDifferentPeriods,n_trials=n_trials)
    if(objective_function=='BasicBifurcation'):
        study.optimize(objective_basic_bifurcation,n_trials=n_trials)
    
    best_trials=study.best_trials
    return best_trials


