#%%
import pickle
import pandas as pd
import numpy as np

from smt.utils import compute_rms_error
from smt.surrogate_models import KRG


try:
    import matplotlib.pyplot as plt
    plot_status = True
except:
    plot_status = False


#%%
"""preprocessing"""
with open(r"220802_data.pkl","rb") as file:
    list_input  = pickle.load(file)
    list_output = pickle.load(file)
    
training_input_data     = []
training_output_data    = []


for initial_position, _output in zip(list_input,list_output):
    for force, load in zip(_output[0],_output[1]):
        _tmp = []
        _tmp.extend(initial_position)
        _tmp.extend([force])
        training_input_data.append(_tmp)
        training_output_data.append(load)

columns = [f'P{i+1}' for i in range(7)]
columns.append('force')
df = pd.DataFrame(training_input_data, columns = columns)

df['load'] = training_output_data

#%%
"""missing values as 1, decrease dimension"""
df = df.astype({"load": str})
df = df.drop(['P3', 'P5', 'P6', 'P7'], axis = 1)
df.replace('inf', np.nan, inplace = True)
df = df.dropna(axis = 0)
df = df.astype({"load": float})

#%%
"""train, test split"""
ndim = 4
training_size = 100
testing_size = 1000

tr_index = np.random.choice(range(len(df)), training_size)
tr_x_column = df.drop('load', axis = 1).columns
tr_y_column = list(df.columns).pop(-1)
tr_x = df[tr_x_column].to_numpy()
tr_y = df[tr_y_column].to_numpy()

te_index = np.random.choice(range(len(df)), testing_size)
te_x_column = df.drop('load', axis = 1).columns
te_y_column = list(df.columns).pop(-1)
# te_x = df[te_x_column].to_numpy()
# te_y = df[te_y_column].to_numpy()
p1 = np.random.choice(np.linspace(20, 40, num=21))
te_x = df[df['P1']==p1][te_x_column].to_numpy()
te_y = df[df['P1']==p1][te_y_column].to_numpy()



#%%
"""Kriging Model"""

# The variable 'theta0' is a list of length ndim.
t = KRG(theta0=[2.18107397e+00, 1.45637443e-06, 9.85017760e-07, 6.74964663e-03],print_prediction = False)
t.set_training_values(tr_x,tr_y)

t.train()

# Prediction of the validation points
y = t.predict_values(te_x)
print('Kriging,  err: '+ str(compute_rms_error(t,te_x,te_y)))
if plot_status:
    
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
    fig = plt.figure()
    plt.plot(te_y, te_y, '-', label='$y_{true}$')
    plt.plot(te_y, y, 'r.', label='$\hat{y}$')
   
    plt.xlabel('$y_{true}$')
    plt.ylabel('$\hat{y}$')
    
    plt.legend(loc='upper left')
    plt.title('Kriging model: validation of the prediction model')   

if plot_status:
    plt.show()

# Value of theta
print("theta values",  t.optimal_theta)
# %%
