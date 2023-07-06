
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter


def tsplot(x, y, percentile_min, percentile_max,x_label,y_label,color, plot_mean=True, 
plot_median=False, line_color='k',line_style='solid',label='',fill=True,label_confidence=""):
    # calculate the lower and upper percentile groups, skipping 50 percentile
    # fill lower and upper percentile groups

    if(fill):
        plt.fill_between(x, percentile_min, percentile_max, alpha=0.2, color='r', label=label_confidence)


    if plot_mean:

        plt.plot(x, np.mean(y, axis=0),linewidth=3,color=color,linestyle=line_style,label=label)


    if plot_median:
        plt.plot(x, np.percentile(y,50,axis=0),linewidth=2,color=color,linestyle=line_style,label=label)
    
    plt.ylabel(y_label,fontsize=25)
    plt.xlabel(x_label,fontsize=25)
    plt.xlim([0,0.4])
    #plt.ylim([0,1])
    plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize=20)

    plt.xticks(fontsize=20)
    plt.legend(fontsize=25)




def rms(x):
    rms=np.sqrt(np.mean(x**2))
    return rms

def determining_seasonal_cycle(data,period,degree):
    X=[i%period for i in range(data.shape[0])]
    
    coef = np.polyfit(X, data, degree)
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value)
    
    return curve

def determining_trend(data,type,degree=2):

    X=[i for i in range(data.shape[0])]
    X = np.reshape(X, (len(X), 1))

    if(type=='linear'):
        model = LinearRegression()
        model.fit(X, data)
        trend = model.predict(X)
    
    if(type=='polynomial'):
        pf = PolynomialFeatures(degree=2)
        Xp = pf.fit_transform(X)
        md2 = LinearRegression()
        md2.fit(Xp, data)
        trend = md2.predict(Xp)

    r2 = r2_score(data, trend)
    rmse = np.sqrt(mean_squared_error(data, trend))
    
    return trend

def plot_power_spectrum(data,sampling_rate=12):
    
    fourier_transform = np.fft.rfft(data)

    abs_fourier_transform = np.abs(fourier_transform)

    power_spectrum = np.square(abs_fourier_transform)

    power_spectrum_normalized = power_spectrum/power_spectrum.max()

    frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))

    fig=plt.figure()
    plt.plot(frequency, power_spectrum_normalized) 
    plt.xlim((0,0.6))
    plt.show()

def save_cdf_as_numpy(path,file_incipit):

    for i in range(1200,1301):

        ds=nc.Dataset(path+"/"+file_incipit+"-"+str(i)+".nc")
        variable=ds['TAUX'][:]
        variable=np.array(variable)
        np.save("wind_anomalies_low/zonal_wind-"+str(i),variable)

def return_classes_weights(data,mean,std,amount_weights):

    classes=np.ones(shape=(data.shape[0]))
    print(mean+1*amount_weights*std)
    classes[(data[:,-1]>mean+3*amount_weights*std)|(data[:,-1]<mean-3*amount_weights*std)]=3
    classes[((data[:,-1]<=mean+3*amount_weights*std) & (data[:,-1]>mean+2*amount_weights*std)) | ((data[:,-1]>=mean-3*amount_weights*std) & (data[:,-1]<mean-2*amount_weights*std))]=2
    classes[((data[:,-1]<=mean+2*amount_weights*std) & (data[:,-1]>mean+1*amount_weights*std)) | ((data[:,-1]>=mean-2*amount_weights*std) & (data[:,-1]<mean-1*amount_weights*std))]=1
    classes[((data[:,-1]<=mean+1*amount_weights*std) & (data[:,-1]>=mean-1*amount_weights*std))]=0

    #classes[(data[:,-1]>mean+3*std)|(data[:,-1]<mean-3*std)]=5
    #classes[((data[:,-1]<=mean+3*std) & (data[:,-1]>=mean+2.5*std)) | ((data[:,-1]>=mean-3*std) & (data[:,-1]<=mean-2.5*std))]=4
    #classes[((data[:,-1]<=mean+2.5*std) & (data[:,-1]>=mean+2*std)) | ((data[:,-1]>=mean-2.5*std) & (data[:,-1]<=mean-2*std))]=4
    #classes[((data[:,-1]<=mean+2*std) & (data[:,-1]>=mean+1.5*std)) | ((data[:,-1]>=mean-2*std) & (data[:,-1]<=mean-1.5*std))]=3    
    #classes[((data[:,-1]<=mean+1.5*std) & (data[:,-1]>=mean+1*std)) | ((data[:,-1]>=mean-1.5*std) & (data[:,-1]<=mean-1*std))]=2
    #classes[((data[:,-1]<=mean+1*std) & (data[:,-1]>=mean+0.5*std)) | ((data[:,-1]>=mean-1*std) & (data[:,-1]<=mean-0.5*std))]=1
    #classes[((data[:,-1]<=mean+0.5*std) & (data[:,-1]>=mean-0.5*std))]=0

    count=Counter(classes)

    weights=np.ones(shape=classes.shape[0])

    for value in count.keys():
        weights[classes==value]=len(weights)/(len(count.keys())*count[value])

    count_weights=Counter(weights)

    return weights


#save_cdf_as_numpy("low_taux_no_season","zonal_wind_no_season")