import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from matplotlib import rc, rcParams

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"




class ZebiakCane:

    def __init__(self,mu,noise):
        self.directory_name="./ZebiakCaneData/resultsMu"+str(mu)+"Noise"+str(noise)
        self.mu=mu
        self.noise=noise


    def load_Zebiak_Cane_data(self,run,monthly=True):

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
            time=[]
            

            for i in range(0,dataset.shape[0],2):
                data_monthly.append(np.mean(dataset[i:i+1,:],axis=0))
                time.append(dataset[i+1,0])
            
            dataset=np.array(data_monthly)
            time=np.array(time)
            
            dataset[:,0]=time
            
            dataset=pd.DataFrame({'time':dataset[:,0],'TW':dataset[:,1],'TE':dataset[:,2],'hW':dataset[:,3],'hE':dataset[:,4]})

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

        
        plt.plot(time,TE_fluct,'-k',linewidth=3)
        plt.xlabel('time [years]',fontsize=25)
        plt.ylabel('TE  [oC]',fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()