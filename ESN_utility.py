from math import tanh
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
import os
from sklearn.linear_model import Ridge
import scipy
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.fft import fft
import netCDF4 as nc
from JinTimmermanUtility import DS
from General_utility import tsplot
from ZebiakCaneUtility import ZebiakCane
from sklearn.preprocessing import StandardScaler
import General_utility
import math
import exploring_Wout
from matplotlib import colors
import random

class ESN:

    def __init__(self,Nx,alpha,density_W,sp,input_variables_number,input_scaling,bias_scaling,regularization,training_weights=1,reservoir_scaling=1):

        self.Nx=Nx
        reservoir_distribution=scipy.stats.uniform(loc=-reservoir_scaling,scale=2*reservoir_scaling).rvs
        self.Win=np.random.uniform(low=-input_scaling,high=input_scaling,size=(Nx,input_variables_number))
        bias=np.random.uniform(low=-bias_scaling,high=bias_scaling,size=(Nx,1))
        self.Win=np.concatenate((bias,self.Win),axis=1)
        W=scipy.sparse.random(Nx,Nx,density=density_W,data_rvs=reservoir_distribution)
        W=np.array(W.A)
        eigenvalues,eigenvectors=np.linalg.eig(W)
        spectral_radius=np.max(np.abs(eigenvalues))
        W=W/spectral_radius
        W=W*sp
        self.W=W
        self.leakage_rate=alpha
        self.input_variables_number=input_variables_number
        self.regularization=regularization
        self.training_weights=training_weights

    def esn_transformation(self,data,warm_up=True):

        time_steps=data.shape[0]
        variables=self.input_variables_number
        x_previous=np.zeros(shape=(self.Nx,1))
        first=True

        warm_up_iterations=10

        if(warm_up):

            for i in range(warm_up_iterations):

                time_step=data[0,:]
                time_step_bias=np.insert(time_step,[0],[1])
                time_step_bias=np.reshape(time_step_bias,newshape=(variables+1,1))
                x_update=np.tanh(np.matmul(self.Win,time_step_bias)+np.matmul(self.W,x_previous))
                x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
                
                x_previous=x_next


        for i in range(time_steps):

            time_step=data[i,:]
            time_step_bias=np.insert(time_step,[0],[1])
            time_step_bias=np.reshape(time_step_bias,newshape=(variables+1,1))

            if((time_step==np.zeros(shape=(self.Nx,1))).all()):

                x_previous=np.zeros(shape=(self.Nx,1))
                i=i+1

                if(i==time_steps):
                    continue

                for j in range(warm_up_iterations):

                    time_step=data[i,:]
                    time_step_bias=np.insert(time_step,[0],[1])
                    time_step_bias=np.reshape(time_step_bias,newshape=(variables+1,1))
                    x_update=np.tanh(np.matmul(self.Win,time_step_bias)+np.matmul(self.W,x_previous))
                    x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
                    
                    x_previous=x_next 

                continue              

            x_update=np.tanh(np.matmul(self.Win,time_step_bias)+np.matmul(self.W,x_previous))
            x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
            
            x_next_final=np.concatenate((time_step_bias,x_next),axis=0)
            
            if first:
                esn_representation=x_next_final.T
                first=False
            else:
                esn_representation=np.concatenate((esn_representation,x_next_final.T),axis=0)
            x_previous=x_next

        return esn_representation
    
    def train(self,data,labels,weights=None):

        self.trainable_model=Ridge(alpha=self.regularization)
        self.trainable_model.fit(data,labels,sample_weight=weights)
        self.W_out=self.trainable_model.coef_

    def evaluate(self,data,labels,weights=None,multioutput=False,pearsons=False):

        if multioutput:

            loss_for_variable=[]

            if(pearsons):

                for variable in range(data.shape[1]):
                
                    loss=pearsonr(labels[:,variable],data[:,variable])
                    loss_for_variable.append(loss.statistic)
            
                return np.mean(loss_for_variable)

            else:

                for variable in range(data.shape[1]):
                    
                    loss=mean_squared_error(labels[:,variable],data[:,variable],sample_weight=weights)
                    root_loss=math.sqrt(loss)
                    std_target=np.std(labels[:,variable])
                    loss_for_variable.append(root_loss/std_target)
                
                return np.mean(loss_for_variable)


        else:

            if(pearsons):
                
                coefficient=pearsonr(labels,data)
                return coefficient.statistic
            
            else:

                return math.sqrt(mean_squared_error(labels,data,sample_weight=weights))


    def predict(self,data):
        
        predictions=self.trainable_model.predict(data)
        return predictions

    def autonomous_evolution(self,scaler,initial_conditions,iterations,warm_up=True,starting_month=1,cycle=False,normalize=False):

        x_previous=np.zeros(shape=(self.Nx,1))
        initial_conditions=np.array(initial_conditions)
        initial_conditions=np.reshape(initial_conditions,(1,initial_conditions.size))
        variables=self.input_variables_number
        time_step=initial_conditions
        first=True
        warm_up_iterations=10

        if(warm_up):

            for i in range(warm_up_iterations):

                time_step_bias=np.insert(time_step,[0],[1])
                time_step_bias=np.reshape(time_step_bias,newshape=(variables+1,1))
                x_update=np.tanh(np.matmul(self.Win,time_step_bias)+np.matmul(self.W,x_previous))
                x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
                
                x_previous=x_next



        for i in range(iterations):

            time_step_bias=np.insert(time_step,[0],[1])
            time_step_bias=np.reshape(time_step_bias,newshape=(variables+1,1))
            x_update=np.tanh(np.matmul(self.Win,time_step_bias)+np.matmul(self.W,x_previous))
            x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
            
            x_next_final=np.concatenate((time_step_bias,x_next),axis=0)

            if(np.isnan(x_next_final.T).any() or np.isinf(x_next_final.T).any() or (x_next_final>10000).any() or (x_next_final<-10000).any()):
                print("not all time series returned")
                return []

#            if(np.max(x_next_final.T)>10):
#                print("not all time series returned")
#                return []
         

            output=self.predict(x_next_final.T)

            if(np.isnan(output).any() or np.isinf(output).any()):
                print("not all time series returned")
                return []
            
            if(cycle):
                month=(starting_month+i)%12
                sin_month=np.sin(2*math.pi*(month)/12)
                cos_month=np.cos(2*math.pi*(month)/12)
                sin_month=np.round(sin_month,1)
                cos_month=np.round(cos_month,1)
                output=np.insert(output,[0],[sin_month,cos_month])
                output=np.reshape(output,newshape=(1,output.shape[0]))


            if(first):

                time_series=output
                first=False
            else:

                time_series=np.concatenate((time_series,output),axis=0)
            
            if(normalize):
                output_scaled=scaler.transform(output)
            
            else:
                output_scaled=output
            x_previous=x_next
            time_step=output_scaled

            

        return time_series

    def plotting_autonumous_evolving_time_series_Jin_Timmerman(self,ds,initial_conditions,steps_esn,
        indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000):

        data=ds.integrate_euler_maruyama(steps,initial_conditions)
        data_monthly=[]

        for i in range(0,data.shape[0],3):
            data_monthly.append(np.mean(data[i:i+3,:],axis=0))

        data_monthly=np.array(data_monthly)
        training_data=data_monthly[steps_skip:,indexes_variables]
        training_labels=training_data[1:,:]
        training_data=training_data[:-1,:]
        training_data_esn_representation=self.esn_transformation(training_data)
        self.train(training_data_esn_representation,training_labels)
        initial_conditions_esn=training_data[0,indexes_variables]
        data_reservoir=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

        x_system=data_reservoir[steps_skip_plotting:,0]

        if(len(indexes_variables)==3):

            y_system=data_reservoir[steps_skip_plotting:,1]
            z_system=data_reservoir[steps_skip_plotting:,2]

            fig=plt.figure(figsize=(12,12))
            ax=plt.subplot(1,1,1)

            plt.plot(ds.dt_monthly*np.arange(data_reservoir[steps_skip_plotting:,:].shape[0]),
            x_system-np.mean(x_system),'-k',linewidth=2.5)
            plt.xlabel('τ',fontsize=25)
            plt.ylabel('x',fontsize=25)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.title("esn output for delta={}".format(ds.delta))


        
        else:

            fig=plt.figure(figsize=(5,5))
            _,ax=plt.subplots()

            plt.plot(ds.dt_monthly*np.arange(data_reservoir[steps_skip_plotting:,:].shape[0]),
            x_system,label=('x oscillation, simulated by reservoir autonumous evolution'))
            plt.xlabel('τ')
            plt.ylabel('x')
            plt.legend(loc='upper right',fontsize=5)


    def return_autonumous_evolving_time_series_Jin_Timmerman(self,ds,initial_conditions,steps_esn,
        indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000):

        data=ds.integrate_euler_maruyama(steps,initial_conditions)
        data_monthly=[]

        for i in range(0,data.shape[0],3):
            data_monthly.append(np.mean(data[i:i+3,:],axis=0))

        data_monthly=np.array(data_monthly)
        training_data=data_monthly[steps_skip:,indexes_variables]
        training_labels=training_data[1:,:]
        training_data=training_data[:-1,:]
        training_data_esn_representation=self.esn_transformation(training_data)
        self.train(training_data_esn_representation,training_labels)
        initial_conditions_esn=training_data[0,indexes_variables]
        data_reservoir=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

        x_system=data_reservoir[steps_skip_plotting:,0]


        return x_system

    def plotting_spectrum_reservoir_different_variables(self,best_parameters,ds,initial_conditions,
    indexes_variables,steps=10000,
    steps_esn=10000,steps_skip=4000,steps_skip_spectrum=4000,iterations=20,normalize=True):

        Sxx=[]

        for i in range(iterations):

            print("run:{}".format(i))
            
            data=ds.integrate_euler_maruyama(steps,initial_conditions)
            data_monthly=[]

            for i in range(0,data.shape[0],3):
                data_monthly.append(np.mean(data[i:i+3,:],axis=0))

            data_monthly=np.array(data_monthly)
            training_data=data_monthly[steps_skip:,indexes_variables]
            training_labels=training_data[1:,:]
            training_data=training_data[:-1,:]
            initial_conditions_esn=training_data[0,:]
            self.__init__(**best_parameters,input_variables_number=len(indexes_variables))
            training_data_esn_representation=self.esn_transformation(training_data)
            self.train(training_data_esn_representation,training_labels)
            reservoir_data=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

            if(reservoir_data==[]):
                continue


            N = reservoir_data[steps_skip_spectrum:,:].shape[0]
            # sample spacing
            T = ds.dt_monthly*N
            xf_tmp = fft(reservoir_data[steps_skip_spectrum:,0]-np.mean(reservoir_data[steps_skip_spectrum:,0]))
            Sxx_tmp=2*ds.dt_monthly**2/T*(xf_tmp*np.conj(xf_tmp))
            Sxx_tmp=Sxx_tmp[:round(reservoir_data[steps_skip_spectrum:,0].shape[0]/2)]
            Sxx_tmp=Sxx_tmp.real

            print(np.max(Sxx_tmp))

            if(normalize):
                
                if(np.round(np.max(Sxx_tmp))!=0):

                    print(np.round(np.max(Sxx_tmp),1))

                    Sxx_tmp=Sxx_tmp/np.max(Sxx_tmp)

            Sxx.append(Sxx_tmp)

        Sxx=np.array(Sxx)
        Sxx_percentile_min=np.percentile(Sxx,5,axis=0)
        Sxx_percentile_max=np.percentile(Sxx,95,axis=0)
        df=1/T
        fNQ=1/ds.dt_monthly/2
        faxis=np.arange(0,fNQ,df)

        tsplot(faxis,Sxx.real,Sxx_percentile_min,Sxx_percentile_max,"frequency [1/τ]","power [normalized]",
        'r',label="mean RC",
        label_confidence="90% confidence RC",line_style='solid',plot_mean=True,plot_median=False)

    def plotting_autonumous_evolving_time_series_Zebiak_Cane(self,zc,run,steps_esn,
        indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000):

        dataset=zc.load_Zebiak_Cane_data(run)
        data_zebiak=np.array(dataset)
        initial_conditions_esn=data_zebiak[steps_skip,indexes_variables]

        training_data=data_zebiak[steps_skip:steps,indexes_variables]
        training_labels=training_data[1:,:]
        training_data=training_data[:-1,:]
        training_data_esn_representation=self.esn_transformation(training_data)
        self.train(training_data_esn_representation,training_labels)
        data_reservoir=self.autonomous_evolution(initial_conditions_esn,steps_esn)

        TE=data_reservoir[steps_skip_plotting:,1]
        hW=data_reservoir[steps_skip_plotting:,2]

        TE_fluct=TE-np.mean(TE)
        hW_fluct=hW-np.mean(hW)

        fig,ax=plt.subplots(2)

        ax[0].plot(0.237*zc.dt*np.arange(data_reservoir[steps_skip_plotting:,:].shape[0]),
        TE_fluct,'.-b',label=('TE oscillation, simulated by reservoir autonumous evolution'))
        ax[0].set_xlabel('years')
        ax[0].set_ylabel('TE anomalies')
        ax[0].legend(loc='upper right',fontsize=5)
        ax[1].plot(hW_fluct,TE_fluct,'.-b',
        label=('Oscillation in hW and TE anomalies plane, simulated by reservoir autonumous evolution'))
        ax[1].set_xlabel('hW')
        ax[1].set_ylabel('TE')
        ax[1].legend(loc='upper right',fontsize=5)
        plt.show()

    def autonumous_evolving_time_series_Zebiak_Cane(self,noise,run,mu_values,steps_esn,
        indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000):
        first=True

        for mu_tmp in mu_values:

            zc=ZebiakCane(mu_tmp,noise,0.35)
            dataset=zc.load_Zebiak_Cane_data(run)
            data_zebiak=np.array(dataset)
            initial_conditions_esn=data_zebiak[steps_skip,indexes_variables]

            training_data=data_zebiak[steps_skip:steps,indexes_variables]
            training_labels=training_data[1:,:]
            training_data=training_data[:-1,:]
            training_data_esn_representation=self.esn_transformation(training_data)
            self.train(training_data_esn_representation,training_labels)
            output=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

            if(output==[]):
                return [],[]

            initial_conditions_esn=output[-1,:]

            if(first):
                data_reservoir=output
                first=False
            else:
                data_reservoir=np.concatenate((data_reservoir,output),axis=0)

        TE=data_reservoir[steps_skip_plotting:,1]
        hW=data_reservoir[steps_skip_plotting:,2]

        TE_fluct=TE-np.mean(TE)
        hW_fluct=hW-np.mean(hW)

        return TE_fluct,hW_fluct,TE



    def plotting_spectrum_autonumous_zebiak(self,zc,indexes_variables,steps=10000,
    steps_esn=10000,steps_skip=4000,steps_skip_spectrum=4000):


        Sxx=[]
        

        for run in os.listdir(zc.directory_name):
            if(run==".DS_Store"):
                continue

            dataset=zc.load_Zebiak_Cane_data(run)
            data=np.array(dataset)
            initial_conditions_esn=data[steps_skip,indexes_variables]


            training_data=data[steps_skip:steps,indexes_variables]
            training_labels=training_data[1:,:]
            training_data=training_data[:-1,:]
            training_data_esn_representation=self.esn_transformation(training_data)
            self.train(training_data_esn_representation,training_labels)
            reservoir_data=self.autonomous_evolution(initial_conditions_esn,steps_esn)
            

            TE=reservoir_data[steps_skip_spectrum:,1]
            print(TE.shape)
            N = TE.shape[0]
            T = zc.dt*N
            xf = fft(TE-np.mean(TE))
            Sxx_tmp=2*zc.dt**2/T*(xf*np.conj(xf))
            print(Sxx_tmp.shape)
            Sxx_tmp=Sxx_tmp[:round(TE.shape[0]/2+0.5)]
            print(Sxx_tmp.shape)
            Sxx.append(np.real(Sxx_tmp))

        Sxx=np.array(Sxx)
        Sxx_percentile_min=np.percentile(Sxx,5,axis=0)
        Sxx_percentile_max=np.percentile(Sxx,95,axis=0)
        df=1/T
        fNQ=1/zc.dt/2
        faxis=np.arange(0,fNQ,df)


        tsplot(faxis,Sxx.real,Sxx_percentile_min,Sxx_percentile_max,label="spectrum TE autonumous evolving Zebiak Cane Delta:{} Noise:{}".format(zc.mu,zc.noise))

    def autonumous_evolving_time_series_CESM_Real_Data(self,data,steps_esn,use_weights=False,cycle=True,weights=None,steps=10000,
    steps_skip=4000):

        scaler=StandardScaler()

        training_data=data[steps_skip:steps,:]

        if(cycle):
            training_labels=training_data[1:,2:]
        else:
            training_labels=training_data[1:,:]

        if(use_weights):
            training_weights=weights[1:]
        
        else:
            training_weights=None


        scaler.fit(training_data)

        training_data=scaler.transform(training_data)
        initial_conditions_esn=training_data[360,:]
        training_data=training_data[:-1,:].copy()
        training_data_esn_representation=self.esn_transformation(training_data)

        if(np.isnan(training_data_esn_representation).any() or np.isinf(training_data_esn_representation).any()):
            print("not all time series returned")
            return []

        self.train(training_data_esn_representation,training_labels,weights=training_weights)
        data_reservoir=self.autonomous_evolution(scaler,initial_conditions_esn,
        steps_esn,warm_up=True,cycle=cycle,normalize=True,starting_month=1)

        if(data_reservoir==[]):
            return []
            
        STE=data_reservoir[:,-1]

        return STE



    def plotting_autonumous_evolving_time_series_basic_bifurcation(self,ds,initial_conditions,steps_esn,
        indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000):

        data=ds.integrate_euler_maruyama(steps,initial_conditions)

        training_data=data[steps_skip:,indexes_variables]
        training_labels=training_data[1:,:]
        training_data=training_data[:-1,:]
        training_data_esn_representation=self.esn_transformation(training_data)
        self.train(training_data_esn_representation,training_labels)
        initial_conditions_esn=training_data[0,indexes_variables]
        data_reservoir=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

        x_system=data_reservoir[steps_skip_plotting:,0]
        y_system=data_reservoir[steps_skip_plotting:,1]

        fig=plt.figure(figsize=(10,7))
        ax=plt.subplot(1,1,1)

        plt.plot(np.arange(data_reservoir[steps_skip_plotting:,:].shape[0]),
        x_system,'-k',linewidth=2)
        plt.xlabel('steps',fontsize=18)
        plt.ylabel('x',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        fig=plt.figure(figsize=(10,7))
        ax=plt.subplot(1,1,1)

        plt.plot(x_system,y_system,'-k',linewidth=2)
        plt.xlabel('x',fontsize=18)
        plt.ylabel('y',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.show()

        U,s,VT=np.linalg.svd(self.trainable_model.coef_,full_matrices=True)

        print(self.trainable_model.coef_.shape)
        print("rank Wout:{}".format(np.linalg.matrix_rank(self.trainable_model.coef_)))
        print("nullity Wout:{}".format(self.trainable_model.coef_.shape[1]-np.linalg.matrix_rank(self.trainable_model.coef_)))

        #U,s,VT=np.linalg.svd(training_data_esn_representation,full_matrices=True)
        #print(training_data_esn_representation.shape)
        #print(s.shape)
        #print(U.shape)
        #print(VT.shape)
        #Sigma = np.zeros((training_data_esn_representation.shape[0], training_data_esn_representation.shape[1]))
        #Sigma[:training_data_esn_representation.shape[1], :training_data_esn_representation.shape[1]] = np.diag(s)
        #n_elements = 5
        #Sigma = Sigma[:, :n_elements]
        #print(Sigma.shape)
        #VT = VT[:n_elements, :]
        #print(VT.shape)
        #T = training_data_esn_representation.dot(VT.T)

        #print(T.shape)

        #plt.plot((T[:,3]-np.mean(T[:,3]))/np.std(T[:,3]),'-k',linewidth=2)
        #plt.plot(T[:,4])
        #plt.xlabel('PC1',fontsize=18)
        #plt.ylabel('PC1',fontsize=18)
        #plt.xticks(fontsize=18)
        #plt.yticks(fontsize=18)


        #plt.plot((training_data[:,3]-np.mean(training_data[:,3]))/np.std(training_data[:,1]))
        #plt.plot(training_data[:,1])
        #plt.show()



    def return_autonumous_evolving_time_series_basic_bifurcation(self,ds,initial_conditions,steps_esn,
        indexes_variables,steps=4000,steps_skip=2000,steps_skip_plotting=5000):

        data=ds.integrate_euler_maruyama(steps,initial_conditions)

        training_data=data[steps_skip:,indexes_variables]
        training_labels=training_data[1:,:]
        training_data=training_data[:-1,:]
        training_data_esn_representation=self.esn_transformation(training_data)
        self.train(training_data_esn_representation,training_labels)
        initial_conditions_esn=training_data[0,indexes_variables]
        data_reservoir=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

        x_system=data_reservoir[steps_skip_plotting:,0]
        y_system=data_reservoir[steps_skip_plotting:,1]


        return x_system,y_system


    def plotting_predictions_basic_bifurcation(self,ds,lead_time,amount_training_data,initial_conditions,steps_skip,
    indexes_predictors,):

        data=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)

        training_data=data[steps_skip:,indexes_predictors]
        training_labels=training_data[6:,:].copy()
        training_data=training_data[:-6,:].copy()
        
        esn_representation_train=self.esn_transformation(training_data)
        
        self.train(esn_representation_train,training_labels)

        data_validation=ds.integrate_euler_maruyama(amount_training_data,initial_conditions=initial_conditions)

        data_validation=np.array(data_validation)
        data_validation=data_validation[steps_skip:,indexes_predictors]
        y_validation=data_validation[6:,:].copy()
        X_validation=data_validation[:-6,:].copy()

        X_validation_esn_representation=self.esn_transformation(X_validation)

        predictions=self.predict(X_validation_esn_representation)


        fig=plt.figure()
        plt.plot(predictions[:,0],label="predictions")
        plt.xlabel("steps")
        plt.ylabel("x")
        plt.plot(y_validation[:,0],label="real values")
        plt.legend()


        fig=plt.figure()
        plt.plot(predictions[:,1],label="predictions")
        plt.xlabel("y")
        plt.ylabel("steps")
        plt.plot(y_validation[:,1],label="real values")
        plt.legend()

    def heat_map_Wout_bb(self,u_values,initial_conditions,steps_esn,
        indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000):

        for u in u_values:

            ds=exploring_Wout.basic_bifurcation(u,1,0.1,0.08)

            data=ds.integrate_euler_maruyama(steps,initial_conditions)

            training_data=data[steps_skip:,indexes_variables]
            training_labels=training_data[1:,:]
            training_data=training_data[:-1,:]
            training_data_esn_representation=self.esn_transformation(training_data)
            self.train(training_data_esn_representation,training_labels)
            initial_conditions_esn=training_data[0,indexes_variables]
            data_reservoir=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

            x_system=data_reservoir[steps_skip_plotting:,0]
            y_system=data_reservoir[steps_skip_plotting:,1]

            fig=plt.figure(figsize=(10,7))
            ax=plt.subplot(1,1,1)

            plt.plot(np.arange(data_reservoir[steps_skip_plotting:,:].shape[0]),
            x_system,'-k',linewidth=2)
            plt.xlabel('steps',fontsize=18)
            plt.ylabel('x',fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.title("Reservoir trajectory for u:{}".format(u),fontsize=18)
            
            fig=plt.figure(figsize=(7,7))
            divnorm=colors.TwoSlopeNorm(vmin=np.min(self.trainable_model.coef_), vcenter=0., 
            vmax=np.max(self.trainable_model.coef_))
            plt.imshow(self.trainable_model.coef_, cmap='coolwarm', interpolation='nearest', norm=divnorm)
 
            # Add colorbar
            plt.colorbar()
            
            plt.title("Heatmap of Wout for u:{}".format(u))

        plt.show()

    def plotting_Wout_time_series(self,u_values,initial_conditions,steps_esn,
        indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000):

        first=True

        for i,u in enumerate(u_values):

            np.random.seed(10)

            ds=exploring_Wout.basic_bifurcation(u,1,0.1,0.08)

            data=ds.integrate_euler_maruyama(steps,initial_conditions)

            training_data=data[steps_skip:,indexes_variables]
            training_labels=training_data[1:,:]
            training_data=training_data[:-1,:]
            training_data_esn_representation=self.esn_transformation(training_data)
            self.train(training_data_esn_representation,training_labels)
            initial_conditions_esn=training_data[0,indexes_variables]
            data_reservoir=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

            x_system=data_reservoir[steps_skip_plotting:,0]
            y_system=data_reservoir[steps_skip_plotting:,1]

            fig=plt.figure(figsize=(10,7))
            ax=plt.subplot(1,1,1)

            plt.plot(np.arange(data_reservoir[steps_skip_plotting:,:].shape[0]),
            x_system,'-k',linewidth=2)
            plt.xlabel('steps',fontsize=18)
            plt.ylabel('x',fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.title("Reservoir trajectory for u:{}".format(u),fontsize=18)
            
            wout=self.trainable_model.coef_

            if(first):
                wout_time_series=np.empty((len(u_values),wout.shape[1]))
                wout_time_series[i,:]=wout[i,:]
                first=False
            
            else:
                wout_time_series[i,:]=wout[i,:]

            fig=plt.figure(figsize=(7,7))

            plt.plot(wout[0,:],label="x components")
            plt.plot(wout[1,:],label="y components")
            plt.legend(fontsize=18)
            
            plt.title("Wout for u:{}".format(u))

        fig=plt.figure(figsize=(7,7))

        for i in range(wout_time_series.shape[0]):

            plt.plot(wout[i,:],label="component x for u:{}".format(u_values[i]))
            plt.title("Wout time_series u:{}".format(u_values))
       
        plt.legend(fontsize=18)


        plt.show()   


    def plotting_mean_time_series(self,u_values,initial_conditions,steps_esn,
            indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000,difference=False):

            internal_iterations=10
            first_external=True

            for i,u in enumerate(u_values):

                first=True

                for j in range(internal_iterations):


                    ds=exploring_Wout.basic_bifurcation(u,1,0.1,0.08)

                    data=ds.integrate_euler_maruyama(steps,initial_conditions)

                    training_data=data[steps_skip:,indexes_variables]
                    training_labels=training_data[1:,:]
                    training_data=training_data[:-1,:]
                    training_data_esn_representation=self.esn_transformation(training_data)

                    self.train(training_data_esn_representation,training_labels)
                    initial_conditions_esn=training_data[0,indexes_variables]
                    data_reservoir=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False)

                    x_system=data_reservoir[steps_skip_plotting:,0]
                    y_system=data_reservoir[steps_skip_plotting:,1]
                    
                    wout=self.trainable_model.coef_

                    if(first):
                        wout_time_series_internal=np.empty((internal_iterations,wout.shape[1]))
                        wout_time_series_internal[j,:]=wout[0,:]
                        first=False
                    
                    else:
                        wout_time_series_internal[j,:]=wout[0,:]

                
                if(first_external):

                    wout_time_series=np.mean(wout_time_series_internal,axis=0)
                    wout_time_series=np.reshape(wout_time_series,(1,wout_time_series.shape[0]))
                    first_external=False

                else:

                    mean=np.mean(wout_time_series_internal,axis=0)
                    mean=np.reshape(mean,newshape=(1,mean.shape[0]))
                    wout_time_series=np.concatenate((wout_time_series,mean),
                    axis=0)

            fig=plt.figure(figsize=(7,7))

            for i in range(wout_time_series.shape[0]):

                plt.plot(wout_time_series[i,:],label="component x for u:{}".format(u_values[i]))
            
            plt.title("Wout mean time_series u:{}".format(u_values))
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
        
            plt.legend(fontsize=18)

            plt.hlines(0,0,46,linestyles=['dashed'],colors='k')

            if(difference):

                diff=np.abs(wout_time_series[0,:]-wout_time_series[1,:])


                f, ax = plt.subplots(figsize = (10,10))
                diff=np.reshape(diff,newshape=(1,diff.shape[0]))
                im=ax.imshow(diff, cmap='coolwarm', interpolation='nearest', aspect=4)
                # Add colorbar
                f.colorbar(im)
                ax.get_yaxis().set_visible(False)
                
                ax.set_title("Heatmap of difference in Wout for u:{} variable x".format(u_values))


            plt.show()
