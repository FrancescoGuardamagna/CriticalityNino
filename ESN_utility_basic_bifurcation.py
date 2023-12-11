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
from sklearn.preprocessing import StandardScaler
import math
import BasicBifurcationUtility
from matplotlib import colors
import random
import autograd.numpy as auto_np
from autograd import jacobian
from matplotlib import rc, rcParams

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"

class ESN:

    def __init__(self,Nx,alpha,density_W,sp,input_variables_number,input_scaling,regularization,bias_scaling=None,training_weights=None,reservoir_scaling=1):

        self.Nx=Nx
        reservoir_distribution=scipy.stats.uniform(loc=-reservoir_scaling,scale=2*reservoir_scaling).rvs
        self.Win=np.random.uniform(low=-input_scaling,high=input_scaling,size=(Nx,input_variables_number))
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

        if(training_weights!=None):
            self.training_weights=training_weights

        self.W_out=None
        self.trained=False

    def __reservoir_one_step_integrator(self,x):

        x=np.reshape(x,newshape=(x.shape[0],1))
        x_update=(1-self.leakage_rate)*x+self.leakage_rate*auto_np.tanh((auto_np.matmul(self.Win,self.W_out)+self.W)@x)
        x_update=np.squeeze(x_update)

        return x_update

    def esn_transformation(self,data,warm_up=True):

        time_steps=data.shape[0]
        variables=self.input_variables_number
        x_previous=np.zeros(shape=(self.Nx,1))
        first=True

        warm_up_iterations=10

        if(warm_up):

            for i in range(warm_up_iterations):

                time_step=data[0,:]
                time_step=np.reshape(time_step,newshape=(variables,1))
                x_update=np.tanh(np.matmul(self.Win,time_step)+np.matmul(self.W,x_previous))
                x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
                
                x_previous=x_next


        for i in range(time_steps):

            time_step=data[i,:]
            
            time_step=np.reshape(time_step,newshape=(variables,1))        

            x_update=np.tanh(np.matmul(self.Win,time_step)+np.matmul(self.W,x_previous))
            x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
            
            x_next_final=x_next
            
            if first:
                esn_representation=x_next_final.T
                first=False
            else:
                esn_representation=np.concatenate((esn_representation,x_next_final.T),axis=0)
            x_previous=x_next

        return esn_representation
    
    def train(self,data,labels,weights=None):

        self.trainable_model=Ridge(alpha=self.regularization,fit_intercept=False)
        self.trainable_model.fit(data,labels,sample_weight=weights)
        self.W_out=self.trainable_model.coef_
        self.trained=True

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

    def autonomous_evolution(self,scaler,initial_conditions,iterations,warm_up=True,
    normalize=False,return_esn_representation=False):

        x_previous=np.zeros(shape=(self.Nx,1))
        initial_conditions=np.array(initial_conditions)
        initial_conditions=np.reshape(initial_conditions,(1,initial_conditions.size))
        variables=self.input_variables_number
        time_step=initial_conditions
        first=True
        warm_up_iterations=10

        if(warm_up):

            for i in range(warm_up_iterations):

                time_step=np.reshape(time_step,newshape=(variables,1))
                x_update=np.tanh(np.matmul(self.Win,time_step)+np.matmul(self.W,x_previous))
                x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
                
                x_previous=x_next



        for i in range(iterations):

            time_step=np.reshape(time_step,newshape=(variables,1))
            x_update=np.tanh(np.matmul(self.Win,time_step)+np.matmul(self.W,x_previous))
            x_next=(1-self.leakage_rate)*x_previous+self.leakage_rate*x_update
            
            x_next_final=x_next

            if(np.isnan(x_next_final.T).any() or np.isinf(x_next_final.T).any() or (x_next_final>10000).any() or (x_next_final<-10000).any()):
                print("not all time series returned")
                return []
         

            output=self.predict(x_next_final.T)

            if(np.isnan(output).any() or np.isinf(output).any()):
                print("not all time series returned")
                return []


            if(first):

                time_series=output
                time_series_esn_space=x_next_final
                first=False

            else:

                time_series=np.concatenate((time_series,output),axis=0)
                time_series_esn_space=np.concatenate((time_series_esn_space,x_next_final),axis=1)
            
            if(normalize):
                output_scaled=scaler.transform(output)
            
            else:
                output_scaled=output

            x_previous=x_next
            time_step=output_scaled

        if(return_esn_representation):

            return time_series,time_series_esn_space    

        return time_series



    def return_autonumous_evolving_time_series_basic_bifurcation(self,ds,initial_conditions,steps_esn,
        indexes_variables,steps=2000,steps_skip=1000,steps_skip_plotting=0):

        data=ds.integrate_euler_maruyama(steps,initial_conditions)

        training_data=data[steps_skip:,indexes_variables]
        training_labels=training_data[1:,:]
        training_data=training_data[:-1,:]
        training_data_esn_representation=self.esn_transformation(training_data)
        self.train(training_data_esn_representation,training_labels)
        initial_conditions_esn=training_data[0,indexes_variables]
        data_reservoir,esn_representation=self.autonomous_evolution(None,initial_conditions_esn,
        steps_esn,normalize=False,return_esn_representation=True)

        x_system=data_reservoir[steps_skip_plotting:,0]
        y_system=data_reservoir[steps_skip_plotting:,1]


        return x_system,y_system,esn_representation,self.W_out

    
    def plotting_autonumous_evolving_time_series_basic_bifurcation_and_training_data(self,ds,initial_conditions,steps_esn,
        indexes_variables,steps=2000,steps_skip=1000,steps_skip_plotting=0):

        data=ds.integrate_euler_maruyama(steps,initial_conditions)

        training_data=data[steps_skip:,indexes_variables]
        training_labels=training_data[1:,:]
        training_data=training_data[:-1,:]
        training_data_esn_representation=self.esn_transformation(training_data)
        self.train(training_data_esn_representation,training_labels)
        initial_conditions_esn=training_data[0,indexes_variables]
        data_reservoir,esn_representation=self.autonomous_evolution(None,initial_conditions_esn,steps_esn,normalize=False,return_esn_representation=True)

        x_system=data_reservoir[steps_skip_plotting:,0]
        y_system=data_reservoir[steps_skip_plotting:,1]

        fig=plt.figure(figsize=(12,12))

        plt.plot(0.1*np.arange(data_reservoir[steps_skip_plotting:,:].shape[0]),
        x_system,linewidth=3,label="self-evolving")
        plt.plot(0.1*np.arange(data_reservoir[steps_skip_plotting:,:].shape[0]),training_data[:,0],c='k',label="training_data")
        plt.xlabel('t',fontsize=25)
        plt.ylabel('x',fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=25,loc='upper left')
        plt.title(u"\u03bc = {}".format(ds.u),fontsize=25,fontweight="bold")

        plt.show()

        return x_system,y_system,esn_representation


