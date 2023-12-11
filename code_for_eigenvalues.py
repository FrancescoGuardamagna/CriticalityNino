from optuna_utility import optuna_optimization
import netCDF4 as nc
from BasicBifurcationUtility import *
from sympy import Function, Matrix, symbols, sin, exp, sqrt, tanh
import ESN_utility_basic_bifurcation as ESN_bb
import numpy as np                 
import matplotlib.pyplot as plt
import wrapper_best_parameters   
from matplotlib import rc, rcParams

rc('axes', linewidth=4)
rc('font', weight='bold')
rcParams['xtick.major.width'] = 3
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.width'] = 3
rcParams['ytick.major.size'] = 10
rcParams["font.weight"] = "bold"
rcParams["axes.labelweight"] = "bold"

wrapper=wrapper_best_parameters()

best_parameters=wrapper.best_parameters_basic_bifurcation

u=0.3
w=1
step=0.1
noise_amplitude=0.08

bb=basic_bifurcation(u,w,step,noise_amplitude)
initial_conditions=[0.1,0.1]

esn=ESN_bb.ESN(**best_parameters,input_variables_number=2)

x_system,y_system,esn_representation=esn.plotting_autonumous_evolving_time_series_basic_bifurcation_and_training_data(bb,initial_conditions,1000,[0,1],2000,1000,0)
esn_representation=np.array(esn_representation)
convergence_point = esn_representation[:,-1]

x = symbols('x0:{}'.format(esn.Nx))

f = Function('f')(Matrix(x))
Win=esn.Win
W=esn.W
alpha=esn.leakage_rate
K=W+Win.dot(esn.W_out)
hyperbolic_matrix=(K*Matrix(x))
tangent_matrix=[tanh(xi) for xi in hyperbolic_matrix]
tangent_matrix=Matrix(tangent_matrix)
indexes=np.logical_and(x_system<=0.1,x_system>=-0.1)
convergence_point = esn_representation[:,-2]

f=(1-alpha)*Matrix(x)+alpha*tangent_matrix

original_space_state=np.concatenate((np.reshape(x_system[-1],newshape=(1,1)),np.reshape(y_system[-1],newshape=(1,1))),axis=0)

Jacobian=f.jacobian(Matrix(x))
Jacobian=Jacobian.subs(zip(x, np.array([0,0,0,0,0,0,0,0])))
Jacobian = np.array(Jacobian, dtype=float)
eigenvalues,eigenvectors=np.linalg.eig(Jacobian)


t = np.linspace(0,np.pi*2,100)

xs = np.real(eigenvalues)
ys = np.imag(eigenvalues)

colors = ['m', 'g', 'r', 'b']
xmin, xmax, ymin, ymax = -1, 1, -1, 1
ticks_frequency = 1


fig= plt.figure(figsize=(12, 12))
plt.plot(np.cos(t), np.sin(t),'k', linewidth=4,label="unit circle")
plt.scatter(xs, ys, linewidths=3, label="eigenvalues")
plt.xlabel("Re",fontsize=25)
plt.ylabel("Im",fontsize=25)
plt.xlim([0,1.2])
plt.ylim([-0.5,0.5])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=25)
plt.show()