import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 
import seaborn as sns
from ESN_utility import ESN
from ZebiakCaneUtility import ZebiakCane
from General_utility import rms
from scipy.signal import find_peaks
from wrapper_best_parameters import wrapper_best_parameters
from matplotlib import rc, rcParams

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"

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

            TE_Rev=TE_Rev[steps_skip:]

            dataset=zc.load_Zebiak_Cane_data("run"+str(i))
            TE_Zc=np.array(dataset['TE'])
            TE_Zc=TE_Zc[steps_skip:amount_training_data]
            TE_Zc=TE_Zc-np.mean(TE_Zc)
            rms_Rev=rms(TE_Rev)
            rms_Zc=rms(TE_Zc)
            index=rms_Rev/rms_Zc
            


            if(rms_Rev>=10):

                print("Reservoir didn't converge for run:{}".format(i))
                continue

            else:
                rms_Rev_list.append(rms_Rev)
                rms_Zc_list.append(rms_Zc)
                index_list.append(index)

            index_list_external.append(np.mean(index_list))
            
    dataframe_dictionary={"\u03BC":zc.mu*np.ones(len(index_list_external)),"C":index_list_external,
    "\u03C3":zc.noise*np.ones(len(index_list_external))}

    return pd.DataFrame(dataframe_dictionary)


def GenerateRmsDataFrame(noise_amplitudes,mu_values,best_params_dictionary,
steps_skip,amount_training_data=1200,iterations=100):

    first=True
    for noise in noise_amplitudes:
        for mu in mu_values:
            zc=ZebiakCane(mu,noise)
            if(first):
                dataset=estimating_rms(zc,iterations,best_params_dictionary[noise],
                amount_training_data,steps_skip)
                first=False
            else:
                dataset=pd.concat([dataset,estimating_rms(zc,iterations,
                best_params_dictionary[noise],amount_training_data,steps_skip)],ignore_index=True)

    dataset.to_csv("ZebiakCaneResults/CriticalIndex{}".format((amount_training_data-steps_skip)/12))

    return dataset


def plot_indexes_Zebiak_Cane(dataset):

    fig=plt.figure(figsize=(12,12))

    b=sns.boxplot(x=dataset['\u03BC'],y=dataset['C'],hue=dataset['\u03C3'],linewidth=3,width=0.8)
    
    b.set_xlabel('\u03BC',fontsize=25)
    b.set_ylabel('C',fontsize=25)
    b.tick_params('both',labelsize=20)
    plt.legend(title="\u03C3",fontsize=25)
    plt.setp(b.get_legend().get_title(), fontsize='25')


    plt.show()


