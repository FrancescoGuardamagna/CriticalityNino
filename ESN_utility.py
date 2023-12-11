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
from General_utility import tsplot
from ZebiakCaneUtility import ZebiakCane
from sklearn.preprocessing import StandardScaler
import General_utility
import math
import BasicBifurcationUtility
from matplotlib import colors
import random
from matplotlib import rc, rcParams
from wrapper_best_parameters import *
from JinTimmermanUtility import *

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"


class ESN:

    def __init__(self,Nx,alpha,density_W,sp,input_variables_number,input_scaling,
    bias_scaling,regularization,training_weights=1,reservoir_scaling=1):

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
    indexes_variables,steps=39600,
    steps_esn=1200,steps_skip=12000,steps_skip_spectrum=0,iterations=20,normalize=True):

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
        if(Sxx.shape[1]!=faxis.shape[0]):
            faxis=faxis[:-1]

        tsplot(faxis,Sxx.real,Sxx_percentile_min,Sxx_percentile_max,"frequency [1/τ]","power [normalized]",
        'r',label="mean RC",
        label_confidence="90% confidence RC",line_style='solid',plot_mean=True,plot_median=False)

    def autonumous_evolving_time_series_Zebiak_Cane(self,noise,run,mu_values,steps_esn,
        indexes_variables,steps=10000,steps_skip=4000,steps_skip_plotting=5000):
        first=True

        for mu_tmp in mu_values:

            zc=ZebiakCane(mu_tmp,noise)
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



