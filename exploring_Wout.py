import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from General_utility import tsplot

class basic_bifurcation:

    def __init__(self,u,w,dt,noise_amplitude):

        self.u=u
        self.w=w
        self.dt=dt
        self.noise_amplitude=noise_amplitude

    def derivative(self,states):

        x=states[0]
        y=states[1]

        derivatives=[(self.u-x**2-y**2)*x+self.w*y,-self.w*x+(self.u-x**2-y**2)*y]

        return derivatives

    def integrate_euler_maruyama(self,steps,initial_conditions):

        solution=[]
        solution.append(initial_conditions)
        x0,y0=initial_conditions

        for i in range(steps):

            derivatives=self.derivative([x0,y0])
            x0=x0+derivatives[0]*self.dt+self.noise_amplitude*np.random.normal(loc=0,scale=1)*np.sqrt(self.dt)
            y0=y0+derivatives[1]*self.dt+self.noise_amplitude*np.random.normal(loc=0,scale=1)*np.sqrt(self.dt)
            solution.append([x0,y0])

        return np.array(solution)

    def plotting(self,initial_conditions,steps=10000,steps_skip=6000):

        solution=self.integrate_euler_maruyama(steps,initial_conditions)

        fig=plt.figure(figsize=(8,5))
        plt.plot(solution[steps_skip:,0],'-k',linewidth=2)
        plt.xlabel("steps",fontsize=18)
        plt.ylabel("x",fontsize=18)



    def plotting_spectrum_dynamical_system(self,initial_conditions,steps=10000,
    steps_skip=1000,iterations=100,normalize=True,fill=True):

        Sxx=[]

        for i in range(iterations):

            data=self.integrate_euler_maruyama(steps,initial_conditions)
            N = data[steps_skip:,:].shape[0]
            # sample spacing
            T = self.dt*N
            xf_tmp = fft(data[steps_skip:,0]-np.mean(data[steps_skip:,0]))
            Sxx_tmp=2*self.dt**2/T*(xf_tmp*np.conj(xf_tmp))
            Sxx_tmp=Sxx_tmp[:round(data[steps_skip:,0].shape[0]/2)]
            Sxx_tmp=Sxx_tmp.real

            if(normalize):
                
                if(np.round(np.max(Sxx_tmp))!=0):

                    Sxx_tmp=Sxx_tmp/np.max(Sxx_tmp)
            Sxx.append(Sxx_tmp)

        Sxx=np.array(Sxx)
        print(Sxx.shape)
        Sxx_percentile_min=np.percentile(Sxx,5,axis=0)
        Sxx_percentile_max=np.percentile(Sxx,95,axis=0)
        df=1/T
        fNQ=1/self.dt/2
        faxis=np.arange(0,fNQ-df,df)


        tsplot(faxis,Sxx.real,Sxx_percentile_min,Sxx_percentile_max,"frequency",
        "power [normalized]",'k',line_style='solid',
        label="",fill=fill)


    