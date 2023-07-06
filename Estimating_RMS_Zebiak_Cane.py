import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 
import seaborn as sns
from ESN_utility import ESN
from ZebiakCaneUtility import ZebiakCane
from General_utility import rms
from scipy.signal import find_peaks

def estimating_rms(zc,iterations,best_params,amount_training_data,steps_skip):

    index_list_external=[]
    damping_ratio_list=[]

    print("estimating rms for mu:{} and noise:{}".format(zc.mu,zc.noise))

    for j in range(iterations):
        esn=ESN(**best_params,input_variables_number=4)
        print("iteration:{}".format(j))
        rms_Rev_list=[]
        rms_Zc_list=[]
        index_list=[]

        for i in range(int(zc.number_of_runs())):
            
            TE_Rev,_,TE_Rev_no_normalized=esn.autonumous_evolving_time_series_Zebiak_Cane(zc.noise,
            "run"+str(i),[zc.mu],2400,[1,2,3,4],amount_training_data,steps_skip,0)
            if(TE_Rev == []):
                continue


            peaks, _ = find_peaks(TE_Rev_no_normalized)
            plt.plot(TE_Rev_no_normalized)
            plt.scatter(peaks,TE_Rev_no_normalized[peaks])
            plt.show()
            peaks=TE_Rev_no_normalized[peaks].copy()
            coefficient=(1/peaks.shape[0])*np.log(peaks[0]/peaks[3])
            print(peaks[0])
            print(peaks[1])
            damping_ratio=coefficient/np.sqrt(4*np.pi**2+coefficient**2)

            print(damping_ratio)

            TE_Rev=TE_Rev[steps_skip:]

            dataset=zc.load_Zebiak_Cane_data("run"+str(i))
            TE_Zc=np.array(dataset['TE'])
            TE_Zc=TE_Zc[steps_skip:amount_training_data]
            TE_Zc=TE_Zc-np.mean(TE_Zc)
            rms_Rev=rms(TE_Rev)
            rms_Zc=rms(TE_Zc)
            index=rms_Rev/rms_Zc
            


            if(rms_Rev>=10):
                continue

            else:
                rms_Rev_list.append(rms_Rev)
                rms_Zc_list.append(rms_Zc)
                index_list.append(index)
                damping_ratio_list.append(damping_ratio)

            index_list_external.append(np.mean(index_list))
            
        dataframe_dictionary={"\u03BC":zc.mu*np.ones(len(index_list_external)),"C":index_list,
        "\u03C3":zc.noise*np.ones(len(index_list_external)),"Zeta":damping_ratio_list}

    return pd.DataFrame(dataframe_dictionary)

def random_evaluation_Rservoir(zc,best_params,iterations,internal_iterations,steps_skip,amount_training_data=2400):

    results=[]

    for iter in range(iterations):
        print("iteration:{}".format(iter))
        random_realization_index=random.sample([i for i in range(zc.number_of_runs())],1)
        index_list=[]

        for i in range(internal_iterations):
            esn=ESN(**best_params,input_variables_number=4)
            TE_Rev,_=esn.autonumous_evolving_time_series_Zebiak_Cane(zc.noise,
            "run"+str(random_realization_index[0]),[zc.mu],2400,[1,2,3,4],
            amount_training_data,steps_skip,0)

            if(TE_Rev==[]):
                continue

            TE_Rev=TE_Rev[steps_skip:]
            rms_Rev=rms(TE_Rev)
            dataset=zc.load_Zebiak_Cane_data("run"+str(random_realization_index[0]))
            TE_Zc=np.array(dataset['TE'])
            TE_Zc=TE_Zc[steps_skip:amount_training_data]
            TE_Zc=TE_Zc-np.mean(TE_Zc)
            rms_Zc=rms(TE_Zc)
            index=rms_Rev/rms_Zc
            index_list.append(index)
        if(np.mean(index_list)>=10):
            continue
        results.append(np.mean(index_list))
    
    results=np.array(results)
    results=np.reshape(results,(results.size,1))
    noise_class=np.full((results.shape[0],1),zc.noise)
    mu_class=np.full((results.shape[0],1),zc.mu)
    results=np.concatenate((results,noise_class,mu_class),axis=1)
    return results

def box_plot_Reservoir_Zebiak_Cane_Data(noise_amplitudes,mu_values,best_params_dictionary,steps_skip,
amount_of_training_data,iterations=100,internal_iterations=3):

    first=True
    for mu_tmp in mu_values:
        for noise in noise_amplitudes:

            print('analyzing mu:{} noise:{}'.format(mu_tmp,noise))
            zc=ZebiakCane(mu_tmp,noise,0.175)
            index_values=random_evaluation_Rservoir(zc,best_params_dictionary[noise],
            iterations=iterations,internal_iterations=internal_iterations,
            steps_skip=steps_skip,amount_training_data=amount_of_training_data)
            if first:
                dataset=index_values
                first=False
            else:
                dataset=np.concatenate((dataset,index_values),axis=0)
    
    DataFrame=pd.DataFrame(dataset,columns=["BifIndex","Noise","Mu"])
    print(DataFrame)
    sns.boxplot(DataFrame,x="Mu",y="BifIndex",hue="Noise")
    plt.show()


def GenerateRmsDataFrame(noise_amplitudes,mu_values,best_params_dictionary,
steps_skip,amount_training_data=2400,iterations=100):

    first=True
    for noise in noise_amplitudes:
        for mu in mu_values:
            zc=ZebiakCane(mu,noise,0.175)
            if(first):
                dataset=estimating_rms(zc,iterations,best_params_dictionary[noise],
                amount_training_data,steps_skip)
                first=False
            else:
                dataset=pd.concat([dataset,estimating_rms(zc,iterations,
                best_params_dictionary[noise],amount_training_data,steps_skip)],ignore_index=True)

    dataset.to_csv("ZebiakCaneResults/IndexDampingRatio60")

    return dataset

def plot_rms_values_and_index(dataset,noise_amplitudes):

    fig,ax=plt.subplots(figsize=(8,8))
    
    for noise in noise_amplitudes:

        dataset_tmp=dataset.loc[dataset['Noise']==noise]
        plt.errorbar(dataset_tmp['Mu'],dataset_tmp['RevMean'],yerr=dataset_tmp['RevStd'],
        capsize=5,capthick=1,marker='o',
        label="rms values Reservoir with standard deviation Noise={}".format(noise))
        plt.errorbar(dataset_tmp['Mu'],dataset_tmp['ZcMean'],yerr=dataset_tmp['ZcStd'],
        capsize=5,capthick=1,marker='o',
        label="rms values ZebiakCane with standard deviation Noise={}".format(noise))
    
    plt.xticks(dataset['Mu'].unique())
    ax.tick_params(labelsize=18)
    plt.axvline(x=3,linestyle='--',color='k',linewidth=2,label="Bifurcation")
    plt.xlabel("µ",fontsize=18)
    plt.ylabel("RMS",fontsize=18)
    plt.legend(loc='center left', bbox_to_anchor=(0.0, 1.01),fontsize=18)

    plt.show()

    fig,ax=plt.subplots(figsize=(7,7))
    
    for noise in noise_amplitudes:

        dataset_tmp=dataset.loc[dataset['Noise']==noise]
        plt.errorbar(dataset_tmp['Mu'],dataset_tmp['Index'],yerr=dataset_tmp['IndexStd'],
        capsize=5,capthick=1,marker='o',
        label="Index values with standard deviation Noise={}".format(noise))
    
    plt.axvline(x=3,linestyle='--',color='k',label="Bifurcation")
    plt.xticks(dataset['Mu'].unique())
    ax.tick_params(labelsize=18)
    plt.xlabel("µ",fontsize=18)
    plt.ylabel("C",fontsize=18)
    plt.legend(loc='center left', bbox_to_anchor=(0.0, 1.01))
  

    plt.show()

def plot_indexes_Zebiak_Cane(dataset):

    fig=plt.figure(figsize=(8,8))

    b=sns.boxplot(x=dataset['\u03BC'],y=dataset['C'],hue=dataset['\u03C3'],linewidth=3,width=0.8)
    
    b.set_xlabel('\u03BC',fontsize=25)
    b.set_ylabel('C',fontsize=25)
    b.tick_params('both',labelsize=20)
    plt.legend(title="\u03C3",fontsize=25)
    plt.setp(b.get_legend().get_title(), fontsize='25')
    


    fig=plt.figure(figsize=(12,12))
    g=sns.boxplot(x=dataset['\u03BC'],y=dataset['Zeta'],hue=dataset['\u03C3'],linewidth=3,width=0.8)

    g.set_xlabel('\u03BC',fontsize=25)
    g.set_ylabel(r'$\zeta$',fontsize=25)
    g.tick_params('both',labelsize=20)
    plt.legend(title="\u03C3",fontsize=25)
    plt.setp(b.get_legend().get_title(), fontsize='25')


    plt.show()