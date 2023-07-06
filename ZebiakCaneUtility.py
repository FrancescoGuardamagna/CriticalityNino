import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.fft import fft
from General_utility import tsplot
import seaborn as sns




class ZebiakCane:

    def __init__(self,mu,noise,dt):
        self.dt=dt
        self.directory_name="./ZebiakCaneData/resultsMu"+str(mu)+"Noise"+str(noise)
        self.mu=mu
        self.noise=noise


    def plotting_spectrum_time_series_Zebiak_Cane(self,steps_skip):


        if(self.noise==0):
    
            dataset=self.load_Zebiak_Cane_data("run1")
            data=np.array(dataset['TE'])
            N = data[steps_skip:].shape[0]
            # sample spacing
            T = self.dt*N
            xf = fft(data[steps_skip:]-np.mean(data[steps_skip:]))
            Sxx=2*self.dt**2/T*(xf*np.conj(xf))
            Sxx=Sxx[:round(data[steps_skip:].shape[0]/2+0.5)]
            Sxx=Sxx.real

            df=1/T
            fNQ=1/self.dt/2
            faxis=np.arange(0,fNQ,df)

            fig=plt.figure()
            plt.plot(faxis,Sxx.real)
            plt.xlabel("frequency")
            plt.ylabel("spectrum")
            plt.xlim([0,0.4])
            plt.show()
        
        else:
            
            Sxx=[]

            for run in os.listdir(self.directory_name):
                if(run==".DS_Store"):
                    continue

                dataset=self.load_Zebiak_Cane_data(run)
                data=np.array(dataset['TE'])
                if(data.shape[0]<1200):
                    data=np.concatenate((data,np.zeros(1200-data.shape[0],)))
                    print(data.shape)
                N = data[steps_skip:].shape[0]
                # sample spacing
                T = self.dt*N
                xf_tmp = fft(data[steps_skip:]-np.mean(data[steps_skip:]))
                Sxx_tmp=2*self.dt**2/T*(xf_tmp*np.conj(xf_tmp))
                Sxx_tmp=Sxx_tmp[:round(data[steps_skip:].shape[0]/2+0.5)]
                Sxx.append(Sxx_tmp.real)

            Sxx=np.array(Sxx)
            print(Sxx.shape)
            Sxx_percentile_min=np.percentile(Sxx,5,axis=0)
            Sxx_percentile_max=np.percentile(Sxx,95,axis=0)
            df=1/T
            fNQ=1/self.dt/2
            faxis=np.arange(0,fNQ,df)
            print(faxis.shape)


            tsplot(faxis,Sxx.real,Sxx_percentile_min,Sxx_percentile_max,label="mean spectrum TE Zebiak Cane Delta:{} Noise:{}".format(self.mu,self.noise))

    def load_Zebiak_Cane_data(self,run,monthly=False):

        file_name=self.directory_name+"/"+run+"/fort.51"

        file=open(file_name,'rt')
        file_modified=open(file_name+"_modified","wt")
        for line in file:
            file_modified.write(line.replace(' -','  -'))
        file.close()
        file_modified.close()
        file_name=file_name+"_modified"
        dataset=pd.read_csv(file_name,names=['time','TW','TE','hW','hE'],skiprows=1,sep='   ',engine='python')
        os.remove(file_name)

        if(monthly):
            dataset=np.array(dataset)

            data_monthly=[]

            for i in range(0,dataset.shape[0],2):
                data_monthly.append(np.mean(dataset[i:i+2,:],axis=0))
            
            dataset=np.array(data_monthly)

        return dataset

    def number_of_runs(self):
        return len(os.listdir(self.directory_name))


    def plot_time_series(self,run,steps_skip_plotting):

        dataset=self.load_Zebiak_Cane_data(run)
        time=dataset['time']
        time=np.array(time)
        time=time[steps_skip_plotting:]
        time=0.237*time
        TW=np.array(dataset['TW'])
        TE=np.array(dataset['TE'])
        TW_mean=np.mean(TW)
        TE_mean=np.mean(TE)
        TW_fluct=TW-TW_mean
        TW_fluct=np.array(TW_fluct)
        TW_fluct=TW_fluct[steps_skip_plotting:]
        TE_fluct=TE-TE_mean
        TE_fluct=np.array(TE_fluct)
        TE_fluct=TE_fluct[steps_skip_plotting:]
        hW=dataset['hW']
        hE=dataset['hE']
        hW_average=np.mean(hW)
        hE_average=np.mean(hE)
        hW_fluct=hW-hW_average
        hE_fluct=hE-hE_average
        hW_fluct=np.array(hW_fluct)
        hE_fluct=np.array(hE_fluct)
        hW_fluct=hW_fluct[steps_skip_plotting:]
        hE_fluct=hE_fluct[steps_skip_plotting:]

        
        plt.plot(TE_fluct,'-k',linewidth=3)
        plt.xlabel('time [years]',fontsize=25)
        plt.ylabel('TE  [oC]',fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

"""
        figure=plt.figure()
        plt.plot(time,TW_fluct,'.-k')
        plt.xlabel('time (years)')
        plt.ylabel('TW  (oC)')


        figure=plt.figure()
        plt.plot(time,hW_fluct,'.-g')
        plt.xlabel('time (years)')
        plt.ylabel('hW (m)')

        figure=plt.figure()
        plt.plot(time,hE_fluct,'.-g')
        plt.xlabel('time (years)')
        plt.ylabel('hE (m)')

        figure=plt.figure()
        plt.plot(TE_fluct,hW_fluct,'.-g')
        plt.title('phase relation between hW anomaly and TE anomaly    T0 = 30 oC')
        plt.xlabel('TE anomaly (C)')
        plt.ylabel('hW anomaly (m)')

"""