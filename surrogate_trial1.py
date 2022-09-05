#%%
import pickle
from this import d
import pandas as pd
import numpy as np

from smt.utils import compute_rms_error
from smt.surrogate_models import KRG, KPLS

 
try:
    import matplotlib.pyplot as plt
    plot_status = True
except:
    plot_status = False


#%%
"""preprocessing"""
with open(r"802_data.pkl","rb") as file: 
    list_input  = pickle.load(file)
    list_output = pickle.load(file)
    
training_input_data     = []
training_output_data    = []


for initial_position, _output in zip(list_input,list_output):
    for force, deformation in zip(_output[0],_output[1]):
        _tmp = []
        _tmp.extend(initial_position)
        _tmp.extend([force])
        training_input_data.append(_tmp)
        training_output_data.append(deformation)

columns = [f'P{i+1}' for i in range(7)]
columns.append('force')
df = pd.DataFrame(training_input_data, columns = columns)

df['deformation'] = training_output_data

#%%
"""missing values as 1, decrease dimension"""
df = df.astype({"deformation": str})
df = df.drop(['P3', 'P5', 'P6', 'P7'], axis = 1)
df.replace('inf', np.nan, inplace = True)
df = df.dropna(axis = 0)
df = df.astype({"deformation": float})



#%%
"""train, test split"""
ndim = 4
training_size = 100
testing_size = 1000

tr_index = np.random.choice(range(len(df)), training_size)
tr_x_column = df.drop('deformation', axis = 1).columns
tr_y_column = list(df.columns).pop(-1)
tr_x = df[tr_x_column].to_numpy()
tr_y = df[tr_y_column].to_numpy()

te_index = np.random.choice(range(len(df)), testing_size)
te_x_column = df.drop('deformation', axis = 1).columns
te_y_column = list(df.columns).pop(-1)

unique_np = df[['P1', 'P2']].to_numpy()
p12 = unique_np[np.random.randint(unique_np.shape[0], size=1), :]

df_alter = df[(df['P1']==p12[0][0]) & (df['P2']==p12[0][1])]
te_x = df_alter[te_x_column].to_numpy()
te_y = df_alter[te_y_column].to_numpy()


#%%
"""Selection of models"""

def model_selection(model = str, theta = [1e-2, 1e-2, 1e-2, 1e-2]):

    if model =='KRG':
        # The variable 'theta0' is a list of length ndim.
        t = KRG(theta0=theta,print_prediction = False)
    elif model == 'KPLS':
        t = KPLS(theta0=theta,print_prediction = False, eval_n_comp = True)

    t.set_training_values(tr_x,tr_y)

    t.train()

    # Prediction of the validation points

    y = t.predict_values(te_x)

    print(model + ' err: '+ str(compute_rms_error(t,te_x,te_y)))

    
        
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(te_y, te_y, '-', label='$y_{true}$')
    ax1.plot(te_y, y, 'r.', label='$\hat{y}$')

    ax1.set_xlabel('$y_{true}$')
    ax1.set_ylabel('response')
    
    ax1.legend(loc='upper left')
    ax1.set_title(model + 'model: validation of the prediction model')  

    ax2.plot(te_x[:, 3], y, 'b-', label = 'prediction of x_test')
    ax2.plot(te_x[:, 3], te_y, 'g--', label = 'true line')
    ax2.set_xlabel('domain')
    ax2.set_ylabel('response')
    ax2.legend(loc = 'upper left')
    plt.show()

    # Value of theta
    return "theta values", t.optimal_theta

#%%
model_selection('KRG')
# model_selection('RMTB', theta = opt_theta)
#%%