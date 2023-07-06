import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from ESN_utility import ESN
from General_utility import rms
from General_utility import determining_seasonal_cycle
from General_utility import tsplot
from wrapper_best_parameters import *
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter
import General_utility
import netCDF4 as nc
from scipy.signal import find_peaks



class CESM:

    def __init__(self,resolution):

        self.resolution=resolution

        self.degree=4

        if(resolution=='high'):
            self.data_no_seasoal=pd.read_csv('./CESM_Data/temperature_dataset_high_resolution_no_seasonal.csv')
            self.data_no_seasonal_moving=self.data_no_seasoal.rolling(3,min_periods=1,center=True).mean()
            self.data_no_seasonal_moving['year']=self.data_no_seasoal['year']
            self.data_no_seasonal_moving['month']=self.data_no_seasoal['month']
            self.data_no_seasonal_moving['cos']=self.data_no_seasoal['cos']
            self.data_no_seasonal_moving['sin']=self.data_no_seasoal['sin']

            for variable in ['TF','ZW','STE']:

                trend=General_utility.determining_trend(self.data_no_seasonal_moving[variable],"linear")
                self.data_no_seasonal_moving[variable]=self.data_no_seasonal_moving[variable].copy()-trend




        if(resolution=='low'):
            self.data_no_seasoal=pd.read_csv('./CESM_Data/temperature_dataset_low_resolution_no_seasonal.csv')
            self.data_no_seasonal_moving=self.data_no_seasoal.rolling(5,min_periods=1,center=True).mean()
            self.data_no_seasonal_moving['year']=self.data_no_seasoal['year']
            self.data_no_seasonal_moving['month']=self.data_no_seasoal['month']


    def plot_data_no_seasonal(self,period):

        interval=period.split('-')
        start_year=int(interval[0])
        end_year=int(interval[1])
        indexes_and=np.logical_and(pd.to_numeric(self.data_no_seasoal['year'])>=start_year, 
        pd.to_numeric(self.data_no_seasoal['year'])<=end_year)
        data=self.data_no_seasoal[indexes_and]
        data_moving=self.data_no_seasonal_moving[indexes_and]

        date=self.data_no_seasonal_moving['year'].astype(str) +"-"+self.data_no_seasonal_moving['month'].astype(str)

        fig=plt.figure(figsize=(16,12))

        plt.plot(date,data_moving['ZW'],'-k',linewidth=3)
        #plt.text(5, 1, f"$R^2$ score EO: {r2_train_EO:0.2f}", style='italic')
        plt.ylabel('Zonal wind stress anomalies [$dyne/cm^2$]',fontsize=30)
        plt.xticks(date[0::240], ha='center', fontsize=25)
        plt.yticks(fontsize=25)



        fig=plt.figure(figsize=(16,12))

        plt.plot(date,data_moving['STE'],'-k',linewidth=3)
        plt.ylabel('Niño 3.4 [K]',fontsize=30)
        plt.xticks(date[0::240], ha='center',fontsize=25)
        plt.yticks(fontsize=25)


        fig=plt.figure(figsize=(16,12))

        plt.plot(date,data_moving['TF'],'-k',linewidth=3)
        plt.ylabel('Thermocline depth anomalies [m]',fontsize=30)
        plt.xticks(date[0::240], ha='center', fontsize=25)
        plt.yticks(fontsize=25)


        plt.show()




    

    def estimating_rms_different_periods(self,periods,internal_iterations,best_params_dictionary,weights=False,test_period="1280-1300",variables=['TF','STE']):

        first=True

        interval=test_period.split('-')
        start_year_test=int(interval[0])
        end_year_test=int(interval[1])
        indexes_and=np.logical_and(pd.to_numeric(self.data_no_seasonal_moving['year'])>=start_year_test, 
        pd.to_numeric(self.data_no_seasonal_moving['year'])<=end_year_test)

        total_data=self.data_no_seasonal_moving.copy()
        total_data=np.array(total_data)
        total_data_for_weights=total_data
        training_mean=np.mean(total_data_for_weights[:,-1])
        training_std=np.std(total_data_for_weights[:,-1])
        total_data=self.data_no_seasonal_moving.copy()
        total_data.drop(total_data.index[np.logical_and(total_data["year"]>start_year_test,total_data["year"]<=end_year_test)],axis=0,inplace=True)

        total_data_for_weights=total_data[variables]
        total_data_for_weights=np.array(total_data_for_weights)

        total_data_for_weights=np.array(total_data_for_weights)
        training_mean=np.mean(total_data_for_weights[:,-1])
        training_std=np.std(total_data_for_weights[:,-1])


        for period in periods:


            print("estimating Index for period:{}".format(period))

            interval=period.split('-')
            start_year=int(interval[0])
            end_year=int(interval[1])
            indexes_and=np.logical_and(pd.to_numeric(total_data['year'])>=start_year, 
            pd.to_numeric(total_data['year'])<=end_year)
            best_params=best_params_dictionary[period]
            total_training_weights=General_utility.return_classes_weights(total_data_for_weights,training_mean,training_std,best_params['training_weights'])
            data=total_data[indexes_and].copy()
            training_weights=total_training_weights[indexes_and]

            data['TF']=data['TF'].copy()/10
            data=np.array(data[variables])
            input_variables_number=len(variables)


            rms_Rev_list=[]
            rms_CESM_list=[]
            index_list=[]
            damping_ratio_list=[]

            column_counter=0

            for i in range(internal_iterations):

                esn=ESN(**best_params,input_variables_number=input_variables_number)
    
                print("iteration:{}".format(i))

                if(weights):
                    STE_Rev=esn.autonumous_evolving_time_series_CESM_Real_Data(data,1200,True,False,training_weights,data.shape[0],0)

                else:
                    STE_Rev=esn.autonumous_evolving_time_series_CESM_Real_Data(data,1200,False,False,None,data.shape[0],0)

                if(STE_Rev == []):
                    continue
 
                STE_CESM=np.array(data[:,-1])
                STE_CESM=STE_CESM-np.mean(STE_CESM)

                STE_Rev=STE_Rev[480:480+STE_CESM.shape[0]]  

                peaks, _ = find_peaks(STE_Rev)

                peaks=STE_Rev[peaks].copy()

                plt.plot(STE_Rev)
                plt.show()
                coefficient=(1/peaks.shape[0])*np.log(peaks[0]/peaks[-1])
                print("coefficient: {}".format(coefficient))
                damping_ratio=coefficient/np.sqrt(4*np.pi**2+coefficient**2)
                print("damping ratio: {}".format(damping_ratio))

                rms_Rev=rms(STE_Rev)
                rms_CESM=rms(STE_CESM)

                index=rms_Rev/rms_CESM

                if(index>2):

                    continue



                rms_Rev_list.append(rms_Rev)
                rms_CESM_list.append(rms_CESM)
                index_list.append(index)
                damping_ratio_list.append(damping_ratio)
                column_counter=column_counter+1



            period_column=np.full(column_counter,period)

            dataframe_dict={'Period':period_column,'RMS_Rev':rms_Rev_list,'RMS_Cesm':rms_CESM_list,
            'Index':index_list,'Damping Ratio':damping_ratio}
        
            if(first):
                results=pd.DataFrame(data=dataframe_dict)
                first=False

            else:
                results=pd.concat([results,pd.DataFrame(data=dataframe_dict)])
        

        fig=plt.figure(figsize=(12,12))

        b=sns.boxplot(results,x="Period",y="Index",linewidth=3,width=0.8)
        
        b.set_xlabel('Period',fontsize=25)
        b.set_ylabel('C',fontsize=25)
        b.tick_params('both',labelsize=20)

        fig=plt.figure(figsize=(12,12))
        g=sns.boxplot(results,x="Period",y="Damping Ratio",linewidth=3,width=0.8)
        g.set_yscale("log")
        g.set_xlabel('Period',fontsize=25)
        g.set_ylabel(r'$\zeta$',fontsize=25)
        g.tick_params('both',labelsize=20)

        results.to_csv("/Users/guard004/Projects/Dynamical System Code Riorganizzato/CesmResults/ResultsIndexesNoWeights")
        


        plt.show()

    
    def compute_mae_CESM(self,periods,weights,test_period="1280-1300",lead_time=9):

        interval=test_period.split('-')
        start_year_test=int(interval[0])
        end_year_test=int(interval[1])
        indexes_and=np.logical_and(pd.to_numeric(self.data_no_seasoal['year'])>=start_year_test, 
        pd.to_numeric(self.data_no_seasoal['year'])<=end_year_test)
        CESM_data=self.data_no_seasonal_moving[indexes_and].copy()

        CESM_ste=CESM_data['STE'].values
        CESM_ste=np.array(CESM_ste)
        CESM_ste=CESM_ste[lead_time:]

        dataset_test=self.data_no_seasonal_moving[indexes_and].copy()
        dataset_test['TF']=dataset_test['TF'].copy()/10

        for period in periods:

            if weights:

                predictions=np.load("CesmResults/predictionsLead9Weights{}.npy".format(period))
                mae=np.mean(np.abs(CESM_ste-predictions))

                print("mae for period with weights:{} is:{}".format(period,mae))

            else:

                predictions=np.load("CesmResults/predictionsLead9NoWeights{}.npy".format(period))
                mae=np.mean(np.abs(CESM_ste-predictions))

                print("mae for period with weights:{} is:{}".format(period,mae))

        




    def plot_predictions(self,train_periods,best_parameters_dictionary,input_variables_number,
    lead_time,iterations,test_period="1280-1300",weights=False,variables=["TF","ZW","STE"],cycle=False):

        if(cycle):
            input_variables_number=input_variables_number+2

        if(len(variables)==3):
            fig1,ax1=plt.subplots(figsize=(15,15))
            fig,ax2=plt.subplots()
            fig,ax3=plt.subplots()

        if(len(variables)==2):
            fig,ax1=plt.subplots(figsize=(12,8))
            fig,ax2=plt.subplots(figsize=(12,8))
        
        interval=test_period.split('-')
        start_year_test=int(interval[0])
        end_year_test=int(interval[1])
        indexes_and=np.logical_and(pd.to_numeric(self.data_no_seasoal['year'])>=start_year_test, 
        pd.to_numeric(self.data_no_seasoal['year'])<=end_year_test)
        CESM_data=self.data_no_seasonal_moving[indexes_and].copy()

        date=self.data_no_seasonal_moving[indexes_and]
        date=date['year'].astype(str) +"-"+date['month'].astype(str)
        date=np.array(date)
        date=date[lead_time:]

        CESM_ste=CESM_data['STE'].values
        CESM_ste=np.array(CESM_ste)
        CESM_ste=CESM_ste[lead_time:]
        Indexes_strong_events = np.where((CESM_ste<-2))

        strong_events_time_series=np.full(CESM_ste.shape,np.nan)
        strong_events_time_series[Indexes_strong_events]=CESM_ste[Indexes_strong_events]

        CESM_wind=CESM_data['ZW']
        CESM_wind=np.array(CESM_wind)
        CESM_wind=CESM_wind[lead_time:]

        CESM_tf=CESM_data['TF']
        CESM_tf=np.array(CESM_tf)
        CESM_tf=CESM_tf[lead_time:]

        dataset_test=self.data_no_seasonal_moving[indexes_and].copy()
        dataset_test['TF']=dataset_test['TF'].copy()/10

        if(cycle):
            data_test=np.array(dataset_test[['sin','cos']+variables])
            data_test=data_test[:-lead_time,:]

        else:
            data_test=np.array(dataset_test[variables])
            data_test=data_test[:-lead_time,:]


        total_data=self.data_no_seasonal_moving.copy()
        total_data.drop(total_data.index[np.logical_and(total_data["year"]>start_year_test,total_data["year"]<=end_year_test)],axis=0,inplace=True)

        if(cycle):
            total_data_for_weights=total_data[['sin','cos']+variables]
        else:
            total_data_for_weights=total_data[variables]

        total_data_for_weights=np.array(total_data_for_weights)
        training_mean=np.mean(total_data_for_weights[:,-1])
        training_std=np.std(total_data_for_weights[:,-1])

        for train_period in train_periods:

            scaler=StandardScaler()

            print("train period:{}".format(train_period))

            best_parameters=best_parameters_dictionary[train_period+"_"+test_period]

            training_weights_total=General_utility.return_classes_weights(total_data_for_weights,training_mean,training_std,best_parameters['training_weights'])

            interval=train_period.split('-')
            start_year=int(interval[0])
            end_year=int(interval[1])
            indexes_and=np.logical_and(pd.to_numeric(self.data_no_seasoal['year'])>=start_year, 
            pd.to_numeric(self.data_no_seasoal['year'])<=end_year)
            indexes_and_weights=np.logical_and(total_data["year"]>=start_year,total_data["year"]<=end_year)
            data_training=self.data_no_seasonal_moving[indexes_and].copy()

            data_training['TF']=data_training['TF'].copy()/10

            if(cycle):

                data_training=np.array(data_training[['sin','cos']+variables])
                training_labels=data_training[lead_time:,2:]

            else:
                data_training=np.array(data_training[variables])
                training_labels=data_training[lead_time:,:]


            
            first_iteration=True

            if(weights):
                training_weights=training_weights_total[indexes_and_weights]
                training_weights=training_weights[lead_time:]

            else:
                training_weights=None

            data_training=data_training[:-lead_time,:]

            scaler.fit(data_training)
            data_training=scaler.transform(data_training)
            data_test_scaled=scaler.transform(data_test,copy=True)

            for i in range(iterations):

                print("iteration:{}".format(i))

                esn=ESN(**best_parameters,input_variables_number=input_variables_number)
                Esn_representation_training=esn.esn_transformation(data_training)

                esn.train(Esn_representation_training,training_labels,weights=training_weights)
                

                Esn_representation_test=esn.esn_transformation(data_test_scaled)
                predictions=esn.predict(Esn_representation_test)

                predictions_ste=predictions[:,-1]
                predictions_ste=predictions_ste[np.newaxis,:]

                if(("ZW" in variables) and ("TF" in variables)):
                    predictions_wind=predictions[:,1]
                    predictions_tf=predictions[:,0]
                    predictions_wind=predictions_wind[np.newaxis,:]
                    predictions_tf=predictions_tf[np.newaxis,:]

                if(("ZW" in variables) and not("TF" in variables)):
                    predictions_wind=predictions[:,0]
                    predictions_wind=predictions_wind[np.newaxis,:]
                
                if(("TF" in variables) and not("ZW" in variables)):
                    predictions_tf=predictions[:,0]
                    predictions_tf=predictions_tf[np.newaxis,:]

                if(first_iteration):

                    predictions_ste_final=predictions_ste

                    if(("ZW" in variables) and ("TF" in variables)):
                        predictions_tf_final=predictions_tf
                        predictions_wind_final=predictions_wind

                    if(("ZW" in variables) and not("TF" in variables)):
                        predictions_wind_final=predictions_wind

                    if(("TF" in variables) and not("ZW" in variables)):
                        predictions_tf_final=predictions_tf
                    
                    first_iteration=False

                else:

                    predictions_ste_final=np.concatenate((predictions_ste_final,predictions_ste),axis=0)

                    if(("ZW" in variables) and ("TF" in variables)):
                        predictions_wind_final=np.concatenate((predictions_wind_final,predictions_wind),axis=0)
                        predictions_tf_final=np.concatenate((predictions_tf_final,predictions_tf),axis=0)

                    if(("ZW" in variables) and not("TF" in variables)):
                        predictions_wind_final=np.concatenate((predictions_wind_final,predictions_wind),axis=0)

                    if(("TF" in variables) and not("ZW" in variables)):
                        predictions_tf_final=np.concatenate((predictions_tf_final,predictions_tf),axis=0)


            ax1.plot(date,np.mean(predictions_ste_final,axis=0),
            label="{}".format(train_period),linewidth=2)
            np.save("CesmResults/predictionsLeadNo9Weights{}".format(train_period),np.mean(predictions_ste_final,axis=0))
            ax1.set_ylabel(ylabel="Niño 3.4 Index",fontsize=25)
            ax1.tick_params(axis='y',labelsize=20)

            mean_ste_predictions=np.mean(predictions_ste_final,axis=0)

            distance_to_extreme_events=np.mean(np.abs(CESM_ste[Indexes_strong_events]-mean_ste_predictions[Indexes_strong_events]))
            print("distance to strong events:{}".format(distance_to_extreme_events))


            if(("ZW" in variables) and ("TF" in variables)):

                ax2.plot(date,np.mean(predictions_wind_final,axis=0),label="train:{} test:{}".format(train_period,test_period))
                ax2.set(ylabel="Zonal wind anomaly")

                ax3.plot(date,10*np.mean(predictions_tf_final,axis=0),label="train:{} test:{}".format(train_period,test_period))
                ax3.set(ylabel="thermocline anomalies over all equatorial area [m]")

            if(("ZW" in variables) and not("TF" in variables)):
                
                ax2.plot(date,np.mean(predictions_wind_final,axis=0),label="train:{} test:{}".format(train_period,test_period))
                ax2.set(ylabel="Zonal wind anomaly")

            if(("TF" in variables) and not("ZW" in variables)):
                
                ax2.plot(date,10*np.mean(predictions_tf_final,axis=0),label="train:{} test:{}".format(train_period,test_period))
                ax2.set(ylabel="thermocline anomalies over all equatorial area [m]")


        font_legend=9

        ax1.plot(date,CESM_ste,'--k',label="CESM",linewidth=4)
        ax1.set_xticks(date[0::60])
        ax1.set_xticklabels(date[0::60],fontsize=20)
        box = ax1.get_position()
        ax1.legend(loc='lower center',bbox_to_anchor=(0.6,0), fontsize=20)
        fig1.savefig("/Users/guard004/Desktop/images_paper/CESMPredictionsNoWeights.png")

        if(("ZW" in variables) and ("TF" in variables)):

            ax2.plot(date,CESM_wind,'--k',label="CESM"+" "+str(self.resolution),linewidth=3)
            ax2.set_xticks(date[0::6])
            ax2.set_xticklabels(date[0::6],rotation=90)
            ax2.grid()
            ax2.legend(fontsize=font_legend)

            ax3.plot(date,CESM_tf,'--k',label="CESM"+" "+str(self.resolution),linewidth=3)
            ax3.set_xticks(date[0::6])
            ax3.set_xticklabels(date[0::6],rotation=90)
            ax3.grid()
            ax3.legend(fontsize=font_legend)

        if(("ZW" in variables) and not("TF" in variables)):

            ax2.plot(date,CESM_wind,'--k',label="CESM"+" "+str(self.resolution),linewidth=3)
            ax2.set_xticks(date[0::6])
            ax2.set_xticklabels(date[0::6],rotation=90)
            ax2.grid()
            ax2.legend(fontsize=font_legend)


        if(("TF" in variables) and not("ZW" in variables)):

            ax2.plot(date,CESM_tf,'--k',label="CESM"+" "+str(self.resolution),linewidth=3)
            ax2.set_xticks(date[0::6])
            ax2.set_xticklabels(date[0::6],rotation=90)
            ax2.grid()
            ax2.legend(fontsize=font_legend)

        plt.show()
        


    def evaluate_performances(self,train_periods,test_period,best_parameters_dictionary,input_variables_number,lead_times,
    iterations,parameters_per_lead_time=False,weights=False,variables=['TF','ZW','STE'],cycle=False):

        fig,ax=plt.subplots()

        if(cycle):
            input_variables_number=input_variables_number+2

        interval=test_period.split('-')
        start_year_test=int(interval[0])
        end_year_test=int(interval[1])
        indexes_and=np.logical_and(pd.to_numeric(self.data_no_seasoal['year'])>=start_year_test, 
        pd.to_numeric(self.data_no_seasoal['year'])<=end_year_test)
        dataset_test=self.data_no_seasonal_moving[indexes_and].copy()
        dataset_test['TF']=dataset_test['TF'].copy()/10

        if(cycle):

            data_test=np.array(dataset_test[['sin','cos']+variables])

        else:

            data_test=np.array(dataset_test[variables])

        

        total_data=self.data_no_seasonal_moving.copy()
        total_data.drop(total_data.index[np.logical_and(total_data["year"]>start_year_test,total_data["year"]<=end_year_test)],axis=0,inplace=True)

        if(cycle):
            total_data_for_weights=total_data[['sin','cos']+variables]
        else:
            total_data_for_weights=total_data[variables]

        total_data_for_weights=np.array(total_data_for_weights)
        training_mean=np.mean(total_data_for_weights[:,-1])
        training_std=np.std(total_data_for_weights[:,-1])
        training_mean=np.mean(total_data_for_weights[:,-1])
        training_std=np.std(total_data_for_weights[:,-1])



        for train_period in train_periods:

            print("evaluating period:{}".format(train_period))
            if(not(parameters_per_lead_time)):
                best_parameters=best_parameters_dictionary[train_period+"_"+test_period]
                total_training_weights=General_utility.return_classes_weights(total_data_for_weights,
                training_mean,training_std,best_parameters['training_weights'])

            interval=train_period.split('-')
            start_year=int(interval[0])
            end_year=int(interval[1])
            indexes_and=np.logical_and(pd.to_numeric(total_data['year'])>=start_year, 
            pd.to_numeric(total_data['year'])<=end_year)
            data_training=total_data[indexes_and].copy()

            data_training['TF']=data_training['TF'].copy()/10

            if(cycle):

                data_training=np.array(data_training[['sin','cos']+variables])

            else:

                data_training=np.array(data_training[variables])

            training_weights=total_training_weights[indexes_and].copy()


            perfomances_per_lead_time_mean=[]
            perfomances_per_lead_time_std=[]

            for lead_time in lead_times:

                print("evaluating lead time:{}".format(lead_time))

                if(parameters_per_lead_time):

                    best_parameters=best_parameters_dictionary[train_period+"_"+test_period+"_"+str(lead_time)]

                else:

                    best_parameters=best_parameters_dictionary[train_period+"_"+test_period]

                if(cycle):

                    training_labels=data_training[lead_time:,2:]
                    test_labels=data_test[lead_time:,2:]
                
                else:

                    training_labels=data_training[lead_time:,:]
                    test_labels=data_test[lead_time:,:]


                if(weights):
                    training_weights_tmp=training_weights[lead_time:]

                else:
                    training_weights_tmp=None
                
                data_training_tmp=data_training[:-lead_time,:]
                data_test_tmp=data_test[:-lead_time,:]

                scaler=StandardScaler()
                scaler.fit(data_training_tmp)
                data_training_tmp=scaler.transform(data_training_tmp)
                data_test_tmp=scaler.transform(data_test_tmp)
                performances=[]

                for i in range(iterations):

                    print("iteration:{}".format(i))

                    esn=ESN(**best_parameters,input_variables_number=input_variables_number)
                    Esn_representation_training=esn.esn_transformation(data_training_tmp)

                    esn.train(Esn_representation_training,training_labels,weights=training_weights_tmp)

                    Esn_representation_test=esn.esn_transformation(data_test_tmp)
                    predictions=esn.predict(Esn_representation_test)

                    result=esn.evaluate(predictions,test_labels,weights=None,multioutput=True)

                    performances.append(result)

                perfomances_per_lead_time_mean.append(np.mean(performances))
                perfomances_per_lead_time_std.append(np.std(performances))

            
            plt.errorbar(lead_times,perfomances_per_lead_time_mean,yerr=perfomances_per_lead_time_std,
            capsize=5,capthick=1,marker='o',
            label="train:{}, test:{}".format(train_period,test_period))
            plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

        
        ax.tick_params(labelsize=10)
        plt.legend()
        plt.xlabel("lead times",fontsize=20)
        plt.ylabel("ACC",fontsize=20)
        plt.grid()
        plt.xticks(lead_times)
        plt.show()








                    





