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
from scipy.fft import fft
import netCDF4 as nc
from numpy import std
from General_utility import tsplot
from pylab import *
from matplotlib import rc, rcParams

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"

class DS:
    
    def __init__(self,dt=0.1,dt_monthly=0.3,amp=0,rho=0.3224,delta=0.2625,
    c=2.3952,k=0.4032,a=8,noise_constant=False,noise_amplitude=0):

        self.amp=amp
        self.delta=delta
        self.rho=rho
        self.noise_amplitude=noise_amplitude
        
        self.c=c
        self.k=k
        self.a=a
        self.std=std
        self.noise_constant=noise_constant
        self.dt=dt
        self.dt_monthly=dt_monthly
    
    def derivative(self,r):
        x=r[0]
        y=r[1]
        z=r[2]

        a=self.a

        if(self.noise_constant):
            Dx=self.noise_amplitude
            Dy=self.noise_amplitude

        else:

            Dx=((1+self.rho*self.delta)*x**2+x*y+self.c*x*(1-tanh(x+z)))*self.noise_amplitude
            Dy=self.rho*self.delta*x**2*self.noise_amplitude

        derivatives=np.array([self.rho*self.delta*(x**2-a*x)+x*(x+y+self.c*(1-tanh(x+z))),
        -self.rho*self.delta*(a*y+x**2),
        self.delta*(self.k-z-(x/2))])

        return derivatives,Dx,Dy


    
    def integrate_euler_maruyama(self,steps,initial_conditions):

        solution=[]
        solution.append(initial_conditions)
        x0,y0,z0=initial_conditions

        for i in range(steps):

            derivatives,Dx,Dy=self.derivative([x0,y0,z0])
            x0=x0+derivatives[0]*self.dt-Dx*np.random.normal(loc=0,scale=1)*np.sqrt(self.dt)
            y0=y0+derivatives[1]*self.dt+Dy*np.random.normal(loc=0,scale=1)*np.sqrt(self.dt)
            z0=z0+derivatives[2]*self.dt
            solution.append([x0,y0,z0])



        return np.array(solution)

    def plotting_time_series(self,initial_conditions,parameter_values,steps,steps_skip):

        first=True
        for delta_tmp in parameter_values:
            ds=DS(a=8,delta=delta_tmp,noise_amplitude=self.noise_amplitude,noise_constant=self.noise_constant)
            data_tmp=ds.integrate_euler_maruyama(steps,initial_conditions)
            if(first):
                data=data_tmp[:,:]
                first=False
            else:
                data=np.concatenate((data,data_tmp),axis=0)
            initial_conditions=data_tmp[-1,:]

        data_monthly=[]

        for i in range(0,data.shape[0],3):
            data_monthly.append(np.mean(data[i:i+3,:],axis=0))

        data_monthly=np.array(data_monthly)
        print(data.shape)
        print(data_monthly.shape)
        x_system=data_monthly[steps_skip:,0]
        y_system=data_monthly[steps_skip:,1]
        z_system=data_monthly[steps_skip:,2]

        #fig,ax=plt.subplots(2)
        #ax[0].plot(self.dt_monthly*np.arange(data_monthly[steps_skip:,:].shape[0]),x_system)
        #ax[0].set_xlabel('tau',fontsize=20)
        #ax[0].set_ylabel('x',fontsize=20)
        #ax[0].tick_params(labelsize=10)
        #ax[1].plot(z_system,x_system)
        #ax[1].set_xlabel('z',fontsize=20)
        #ax[1].set_ylabel('x',fontsize=20)
        #ax[1].tick_params(labelsize=10)

        fig,ax=plt.subplots(figsize=(10, 10))
        plt.plot(self.dt_monthly*np.arange(data_monthly[steps_skip:,:].shape[0]),x_system,'-k',linewidth=2.5)
        plt.xlabel('τ',fontsize=22)
        plt.ylabel('x',fontsize=22)

        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
       

        plt.show()

    def plotting_spectrum_dynamical_system(self,initial_conditions,steps=10000,
    steps_skip=1000,iterations=100,normalize=True,fill=True,line_style='solid'):

        Sxx=[]

        for i in range(iterations):

            data=self.integrate_euler_maruyama(steps,initial_conditions)
            data_monthly=[]

            for i in range(0,data.shape[0],3):
                data_monthly.append(np.mean(data[i:i+3,:],axis=0))

            data_monthly=np.array(data_monthly)
            N = data_monthly[steps_skip:,:].shape[0]
            # sample spacing
            T = self.dt_monthly*N
            xf_tmp = fft(data_monthly[steps_skip:,0]-np.mean(data_monthly[steps_skip:,0]))
            Sxx_tmp=2*self.dt_monthly**2/T*(xf_tmp*np.conj(xf_tmp))
            Sxx_tmp=Sxx_tmp[:round(data_monthly[steps_skip:,0].shape[0]/2)]
            Sxx_tmp=Sxx_tmp.real

            if(normalize):
                
                if(np.round(np.max(Sxx_tmp))!=0):

                    Sxx_tmp=Sxx_tmp/np.max(Sxx_tmp)
            
            print(np.max(Sxx_tmp))
            Sxx.append(Sxx_tmp)

        Sxx=np.array(Sxx)
        print(Sxx.shape)
        Sxx_percentile_min=np.percentile(Sxx,5,axis=0)
        Sxx_percentile_max=np.percentile(Sxx,95,axis=0)
        df=1/T
        fNQ=1/self.dt_monthly/2
        faxis=np.arange(0,fNQ-df,df)

        tsplot(faxis,Sxx.real,Sxx_percentile_min,Sxx_percentile_max,"frequency [1/τ]",
        "power [normalized]",'k',line_style=line_style,
        label="deterministic JT",fill=fill)



