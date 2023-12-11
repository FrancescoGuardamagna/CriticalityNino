import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ESN_utility import ESN
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from General_utility import determining_trend
from General_utility import rms
from scipy import signal
from scipy.stats import pearsonr
import General_utility
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import math
from wrapper_best_parameters import *


class RealData:

    def __init__(self):

        self.data=pd.read_csv('./RealData/real_data_dataframe_new.csv')
        self.data['TF']=signal.detrend(self.data['TF'])
        self.data['STE']=signal.detrend(self.data['STE'])
        self.data['ZW']=signal.detrend(self.data['ZW'])

        self.data['TF']=self.data['TF'].rolling(3,min_periods=1).mean()
        self.data['STE']=self.data['STE'].rolling(3,min_periods=1).mean()
        self.data['ZW']=self.data['ZW'].rolling(3,min_periods=1).mean()

        self.data=self.data.dropna()


            
    def plot_data_no_seasonal(self,period):

        interval=period.split('-')
        start_year=int(interval[0])
        end_year=int(interval[1])
        indexes_and=np.logical_and(pd.to_numeric(self.data['year'])>=start_year, 
        pd.to_numeric(self.data['year'])<=end_year)
        data_moving=self.data[indexes_and]

        date=self.data['year'].astype(str) +"-"+self.data['month'].astype(str)

        fig=plt.figure(figsize=(16,12))

        plt.plot(date,data_moving['ZW'],'k',linewidth=3)
        plt.ylabel('Zonal wind stress anomalies [m\u00b2/s\u00b2]',fontsize=30)
        plt.xticks(date[0::120],fontsize=25)
        plt.yticks(fontsize=25)



        fig=plt.figure(figsize=(16,12))

        plt.plot(date,data_moving['STE'],'k',linewidth=3)
        plt.ylabel('Niño 3.4 [K]',fontsize=30)
        plt.xticks(date[0::120],fontsize=25)
        plt.yticks(fontsize=25)


        fig=plt.figure(figsize=(16,12))

        plt.plot(date,data_moving['TF'],'k',linewidth=3)
        plt.ylabel('Thermocline depth anomalies [m]',fontsize=30)
        plt.xticks(date[0::120],fontsize=25)
        plt.yticks(fontsize=25)


        plt.show()


    def plot_predictions(self,train_periods,best_parameters_dictionary,input_variables_number,
    lead_time,iterations,test_period="2000-2020",weights=True,variables=["TF","ZW","STE"]):

        if(len(variables)==3):
            fig1,ax1=plt.subplots(figsize=(12,12))
            fig,ax2=plt.subplots()
            fig,ax3=plt.subplots()

        if(len(variables)==2):
            fig1,ax1=plt.subplots()
            fig,ax2=plt.subplots()
        
        interval=test_period.split('-')
        start_year_test=int(interval[0])
        end_year_test=int(interval[1])
        indexes_and=np.logical_and(pd.to_numeric(self.data['year'])>=start_year_test, 
        pd.to_numeric(self.data['year'])<=end_year_test)
        Real_data=self.data[indexes_and].copy()

        Real_ste=Real_data['STE'].values
        Real_ste=np.array(Real_ste)
        Real_ste=Real_ste[lead_time:]
        Indexes_strong_events_positive = np.where((Real_ste>1))
        strong_events_positive=np.full(shape=Real_ste.shape,fill_value=np.nan)
        strong_events_positive[Indexes_strong_events_positive]=Real_ste[Indexes_strong_events_positive]

        Indexes_strong_events_negative = np.where((Real_ste<-1))
        strong_events_negative=np.full(shape=Real_ste.shape,fill_value=np.nan)
        strong_events_negative[Indexes_strong_events_negative]=Real_ste[Indexes_strong_events_negative]



        Real_wind=Real_data['ZW']
        Real_wind=np.array(Real_wind)
        Real_wind=Real_wind[lead_time:]

        Real_tf=Real_data['TF']
        Real_tf=np.array(Real_tf)
        Real_tf=Real_tf[lead_time:]

        date=self.data[indexes_and]
        date=date['year'].astype(str) +"-"+date['month'].astype(str)
        date=np.array(date)
        date=date[lead_time:]

        dataset_test=self.data[indexes_and].copy()
        dataset_test['TF']=dataset_test['TF'].copy()/10
        data_test=np.array(dataset_test[variables])
        data_test=data_test[:-lead_time,:]


        total_data=self.data.copy()
        total_data.drop(total_data.index[np.logical_and(total_data["year"]>start_year_test,total_data["year"]<=end_year_test)],axis=0,inplace=True)


        total_data_for_weights=total_data[['sin','cos']+variables]
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
            indexes_and=np.logical_and(pd.to_numeric(self.data['year'])>=start_year, 
            pd.to_numeric(self.data['year'])<=end_year)
            indexes_and_weights=np.logical_and(total_data["year"]>=start_year,total_data["year"]<=end_year)
            data_training=self.data[indexes_and].copy()

            data_training['TF']=data_training['TF'].copy()/10

            
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
            label="RC",linewidth=3)
            ax1.set_ylabel("Niño 3.4 Index [K]",fontsize=20)

            mean_ste_predictions=np.mean(predictions_ste_final,axis=0)

            np.save("ResultsReal/predictionsLead{}NoWeights".format(lead_time),np.mean(predictions_ste_final,axis=0))

            distance_to_extreme_events_positive=np.mean(np.abs(Real_ste[Indexes_strong_events_positive]-mean_ste_predictions[Indexes_strong_events_positive]))
            print("distance to strong events Nino:{}".format(distance_to_extreme_events_positive))

            distance_to_extreme_events_negative=np.mean(np.abs(Real_ste[Indexes_strong_events_negative]-mean_ste_predictions[Indexes_strong_events_negative]))
            print("distance to strong events Nina:{}".format(distance_to_extreme_events_negative)) 

            distance_all_test=np.mean(np.abs(Real_ste-mean_ste_predictions))    
            print("distance all period:{}".format(distance_all_test))      

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



        ax1.plot(date,Real_ste,'--k',label="Real",linewidth=4)
        ax1.set_xticks(date[0::48])
        ax1.set_xticklabels(date[0::48],fontsize=20)
        ax1.tick_params(axis='y',labelsize=25)
        ax1.tick_params(axis='x',labelsize=25)
        ax1.legend(loc="upper left",fontsize=25)

        if(("ZW" in variables) and ("TF" in variables)):

            ax2.plot(date,Real_wind,'--k',label="Real Observations",linewidth=3)
            ax2.set_xticks(date[0::6])
            ax2.set_xticklabels(date[0::6],rotation=90)
            ax2.grid()
            ax2.legend()

            ax3.plot(date,Real_tf,'--k',label="Real Observations",linewidth=3)
            ax3.set_xticks(date[0::6])
            ax3.set_xticklabels(date[0::6],rotation=90)
            ax3.grid()
            ax3.legend()

        if(("ZW" in variables) and not("TF" in variables)):

            ax2.plot(date,Real_wind,'--k',label="Real Observations",linewidth=3)
            ax2.set_xticks(date[0::6])
            ax2.set_xticklabels(date[0::6],rotation=90)
            ax2.grid()
            ax2.legend()


        if(("TF" in variables) and not("ZW" in variables)):

            ax2.plot(date,Real_tf,'--k',label="Real Observations",linewidth=3)
            ax2.set_xticks(date[0::6])
            ax2.set_xticklabels(date[0::6],rotation=90)
            ax2.grid()
            ax2.legend()


    def estimating_rms_different_periods(self,internal_iterations,best_params_dictionary,
    variables=['TF','ZW','STE'],test_period="2000-2020",weights=False,lead_times=[3,6,9],period="1960-2000"):

        first=True

        input_variables_number=len(variables)

        interval=test_period.split('-')
        start_year_test=int(interval[0])
        end_year_test=int(interval[1])
        indexes_and=np.logical_and(pd.to_numeric(self.data['year'])>=start_year_test, 
        pd.to_numeric(self.data['year'])<=end_year_test)

        total_data=self.data.copy()
        total_data.drop(total_data.index[np.logical_and(total_data["year"]>start_year_test,total_data["year"]<=end_year_test)],axis=0,inplace=True)

        total_data_for_weights=total_data[['sin','cos']+variables]
        total_data_for_weights=np.array(total_data_for_weights)

        total_data_for_weights=np.array(total_data_for_weights)
        training_mean=np.mean(total_data_for_weights[:,-1])
        training_std=np.std(total_data_for_weights[:,-1])


        for lead_time in lead_times:

            print("estimating Index for lead_time:{}".format(lead_time))

            interval=period.split('-')
            start_year=int(interval[0])
            end_year=int(interval[1])
            indexes_and=np.logical_and(pd.to_numeric(self.data['year'])>=start_year, 
            pd.to_numeric(self.data['year'])<=end_year)
            data=self.data[indexes_and].copy()
            best_params=best_params_dictionary[lead_time]
            total_training_weights=General_utility.return_classes_weights(total_data_for_weights,training_mean,training_std,best_params['training_weights'])
            indexes_and_weights=np.logical_and(total_data["year"]>=start_year,total_data["year"]<=end_year)
            data=self.data[indexes_and].copy()
            training_weights=total_training_weights[indexes_and_weights]

            data['TF']=data['TF'].copy()/10

            
            data=np.array(data[variables])


            rms_Rev_list=[]
            rms_Real_list=[]
            damping_ratio_list=[]
            index_list=[]


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
 
                STE_Real=np.array(data[:,input_variables_number-1])
                STE_Real=STE_Real-np.mean(STE_Real)
                STE_Rev=STE_Rev-np.mean(STE_Rev)   

              



                peaks, _ = find_peaks(STE_Rev)

                if(len(peaks)<5):
                    continue

                
                peaks=STE_Rev[peaks].copy()

                
                coefficient=(1/peaks.shape[0])*np.log(np.abs(peaks[0]/peaks[4]))
                damping_ratio=coefficient/np.sqrt(4*np.pi**2+coefficient**2)

                STE_Rev=STE_Rev[720:]

                rms_Rev=rms(STE_Rev)
                rms_Real=rms(STE_Real)

                index=rms_Rev/rms_Real

                if(index>2 or math.isnan(index) or math.isnan(coefficient)):

                    continue

                damping_ratio=abs(damping_ratio)

                if(coefficient==0):
                    continue

                rms_Rev_list.append(rms_Rev)
                rms_Real_list.append(rms_Real)
                index_list.append(index)
                damping_ratio_list.append(damping_ratio)
                column_counter=column_counter+1

            



            lead_column=np.full(column_counter,lead_time)
            period_colums=np.full(column_counter,period)

            dataframe_dict={'Period':period_colums,'Lead_time':lead_column,'RMS_Rev':rms_Rev_list,'RMS_Real':rms_Real_list,
            'Index':index_list,'Damping Ratio':damping_ratio_list}
        
            if(first):
                results=pd.DataFrame(data=dataframe_dict)
                first=False

            else:
                results=pd.concat([results,pd.DataFrame(data=dataframe_dict)])

        fig=plt.figure(figsize=(13,13))

        b=sns.boxplot(results,x="Period",y="Index",hue="Lead_time",linewidth=3,width=0.8)
        
        b.set_xlabel('Period',fontsize=25)
        b.set_ylabel('C',fontsize=25)
        b.tick_params('both',labelsize=20)

        plt.legend(title="Lead",fontsize=20)
        plt.setp(b.get_legend().get_title(), fontsize='25')
        

        fig=plt.figure(figsize=(13,13))

        g=sns.boxplot(results,x="Period",y="Damping Ratio",hue="Lead_time",linewidth=3,width=0.8)
        g.set_xlabel('Period',fontsize=25)
        g.set_ylabel(r'$\zeta$',fontsize=25)
        g.tick_params('both',labelsize=20)


        plt.legend(title="Lead",fontsize=20)
        plt.setp(g.get_legend().get_title(), fontsize='25')
        results.to_csv("ResultsReal/ResultsIndexes")

        plt.show()