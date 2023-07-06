from optuna_utility import optuna_optimization
from CESM_utility import CESM
from RealData_utility import *
import General_utility
from JinTimmermanUtility import *
from Estimating_RMS_Zebiak_Cane import *
from wrapper_best_parameters import * 
import netCDF4 as nc
from exploring_Wout import *
from sympy import Function, Matrix, symbols, sin, exp, sqrt, tanh
import ESN_utility_basic_bifurcation as ESN_bb

wrapper=wrapper_best_parameters()

best_parameters=wrapper.best_parameters_basic_bifurcation

bb=basic_bifurcation(-0.3,1,0.1,0.08)
initial_conditions=[0.1,0.1]

esn=ESN_bb.ESN(**best_parameters,input_variables_number=2)

fig=plt.figure(figsize=(12,12))
bb.plotting(initial_conditions,2000,1000)

esn.plotting_autonumous_evolving_time_series_basic_bifurcation(bb,initial_conditions,1000,[0,1],2000,1000,0)

plt.show()

mse_x_list=[]
mse_y_list=[]

#for i in range(100):

#    mse_x,mse_y=esn.plotting_predictions_basic_bifurcation(bb,6,2000,initial_conditions,1000,[0,1])
#    mse_x_list.append(mse_x)
#    mse_y_list.append(mse_y)

#print("mean mse x:{}".format(np.mean(mse_x_list)))
#print("mean mse y:{}".format(np.mean(mse_y_list)))




#esn.plotting_predictions_basic_bifurcation(bb,6,2000,
#initial_conditions,1000,[0,1])

autonoumos_time_series_x,autonoumos_time_series_y,esn_representation,W_out=esn.return_autonumous_evolving_time_series_basic_bifurcation(bb,
initial_conditions,2000,[0,1],steps_skip_plotting=0)

esn_representation=np.array(esn_representation)

x = symbols('x0:{}'.format(esn.Nx))

# Define the function
f = Function('f')(Matrix(x))
Win=esn.Win
print(Win.shape)
W=esn.W
print(W.shape)
alpha=0.3
#print((1-alpha)*Matrix(x))
K=W+Win.dot(W_out)
#print(K*Matrix(x))
hyperbolic_matrix=(alpha*(K*Matrix(x)))
tangent_matrix=[tanh(xi) for xi in hyperbolic_matrix]
tangent_matrix=Matrix(tangent_matrix)

#print(tangent_matrix)
convergence_point = esn_representation[:,-1]
print(esn_representation[:,-1])

f=(1-alpha)*Matrix(x)+tangent_matrix
Jacobian=f.jacobian(Matrix(x))
Jacobian=Jacobian.subs(zip(x, convergence_point))
Jacobian = np.array(Jacobian, dtype=float)
eigenvalues,eigenvectors=np.linalg.eig(Jacobian)
print(eigenvalues)

#esn.plotting_mean_time_series([-0.4,0.4],initial_conditions,2000,[0,1],
#steps_skip_plotting=0,difference=True)


import numpy as np                 # v 1.19.2
import matplotlib.pyplot as plt    # v 3.3.2

t = np.linspace(0,np.pi*2,100)


# Enter x and y coordinates of points and colors
xs = np.real(eigenvalues)
ys = np.imag(eigenvalues)

colors = ['m', 'g', 'r', 'b']

# Select length of axes and the space between tick labels
xmin, xmax, ymin, ymax = -1, 1, -1, 1
ticks_frequency = 1

# Plot points
fig= plt.figure(figsize=(12, 12))
plt.plot(np.cos(t), np.sin(t),'k', linewidth=4,label="unit circle")
plt.scatter(xs, ys, linewidths=3, label="eigenvalues")
plt.xlabel("Re",fontsize=25)
plt.ylabel("Im",fontsize=25)
plt.ylim([-0.25,0.25])
plt.xlim([0.5,1.2])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=25)