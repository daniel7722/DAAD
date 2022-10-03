# In[1]:
"""importing necessary functions"""
import sys
#sys.path.insert(0, r'C:\LokaleDaten\s8913814\Documents\Python_experimental')
from _220302_data_creation_for_NN_ADDITIONAL_FUNCTIONS import *
import random
from sklearn.metrics import mean_squared_error
from smt.utils import compute_rms_error
from smt.surrogate_models import KRG, KPLS
from random import randint


# In[2]:
"""reading of RAW data and preprocessing"""

#read RAW data
path_data=r'/Users/danielhuang/coding/DAAD_main/Data/220602_RAW_data_population_WITHOUT_COMMENT'
list_chromosome, list_fitness, coords_RAW = read_RAW_data(path=path_data)

#create the distributed points
number_of_points = 50 #too little and you loose information about the fea path
coords_prepro = get_distributed_points_of_desired_amount(
    coords_RAW=coords_RAW,
    number_of_points=number_of_points
    )

#delete elements that contain erroneously values of nan (due to division by zero)
list_chromosome, list_fitness, coords_RAW, coords_prepro = delete_results_with_nan(
    list_chromosome=list_chromosome,
    list_fitness=list_fitness,
    coords_RAW=coords_RAW,
    coords_prepro=coords_prepro
    )
#delete elements that erroneously don't have the desired number of points (don't know why this sometimes doesn't work)
list_chromosome, list_fitness, coords_RAW, coords_prepro = delete_results_of_false_length(
    list_chromosome=list_chromosome,
    list_fitness=list_fitness,
    coords_RAW=coords_RAW,
    coords_prepro=coords_prepro,
    number_of_points=number_of_points
    )

#reduced the chromosome to the first 8 entries (the coords of the mechnisms edge nodes)
list_chromosome = simplify_chromosome(list_chromosome=list_chromosome)

#visualize the RAW and preprocessed coordinates
# path_pics=r".\Pics"
# index = 2
# compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
#                                         coords_RAW_single    = coords_RAW[index],
#                                         coords_prepro_single = coords_prepro[index],
#                                         title                = str(index)
#                                         )

#np.save("input_X.npy",list_chromosome)
#np.save("output_Y.npy",coords_prepro)

# In[3]:
"""benchmarking"""
# coords_prepro_shuffled = coords_prepro.copy()
# random.shuffle(coords_prepro_shuffled)

# mse_benchmark = mean_squared_error(coords_prepro, coords_prepro_shuffled)
# #me = mse_benchmark**0.5
# print(mse_benchmark)

# In[4]:
"""Data Splitting"""

# Taking a portion of data becasue too many data entries crash jupyter kernel
testing_data1 = np.array(coords_prepro[:100])
testing_data2 = np.array(list_chromosome[:100])
training_data1 = np.array(coords_prepro[100:200])
training_data2 = np.array(list_chromosome[100:200])

# separating training set and testing set
transpose_testing = np.array(testing_data1).T
data_x_test = transpose_testing[:50]
data_y_test = transpose_testing[50:]
transpose_training = np.array(training_data1).T
data_x_train = transpose_training[:50]
data_y_train = transpose_training[50:]

# cleaning and recreating data that is usable as I wish
# testing data
data_process1     = []
for index, i in enumerate(testing_data2):
    for indexs, (j, z) in enumerate(zip(data_x_test, data_y_test)):
        _temp = []
        _temp.extend(i)
        _temp.extend([indexs])
        _temp.extend([j[index]])
        _temp.extend([z[index]])
        data_process1.append(_temp)

# training data
data_process2     = []
for index, i in enumerate(training_data2):
    for indexs, (j, z) in enumerate(zip(data_x_train, data_y_train)):
        _temp = []
        _temp.extend(i)
        _temp.extend([indexs])
        _temp.extend([j[index]])
        _temp.extend([z[index]])
        data_process2.append(_temp)

# switching to DataFrame
columns =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
df_process1 = pd.DataFrame(data_process1, columns= columns)
df_process2 = pd.DataFrame(data_process2, columns= columns)

# generating testing DataFrame (one configuration)
unique_np = df_process1[['0', '1', '2']].to_numpy()
p124 = unique_np[np.random.randint(unique_np.shape[0], size=1), :]
df_alter = df_process1[(df_process1['0']==p124[0][0]) & (df_process1['1']==p124[0][1]) & (df_process1['2']==p124[0][2])]

# separating it into input and two outputs
x_test = df_alter[['0', '1', '2', '3', '4', '5', '6', '7', '8']].to_numpy()
y_test1 = df_alter['9'].to_numpy()
y_test2 = df_alter['10'].to_numpy()

# randomly generate index number of training set
foo = []
for i in range(2000):
    foo.append(randint(0, len(df_process2)-1))
foo = list(set(foo))

# separating it into input and two outputs
x_train = df_process2.iloc[foo, :9].to_numpy()
y_train1 = df_process2.iloc[foo, 9].to_numpy()
y_train2 = df_process2.iloc[foo, 10].to_numpy()

 # In[5]:
#Data scaling
"""      plotting not yet working in combination with scaled data
inp_scaler = StandardScaler()
x_train = inp_scaler.fit_transform(x_train)
x_test = inp_scaler.transform(x_test)
x_validation = inp_scaler.transform(x_validation)

out_scaler = StandardScaler()
y_train = out_scaler.fit_transform(y_train)#.reshape(-1, 1)).reshape(y_train.shape)
y_test = out_scaler.transform(y_test)#.reshape(-1, 1)).reshape(y_test.shape)
y_validation = out_scaler.transform(y_validation)
"""

# In[6]:
"""Training model 1"""
# model 1 is for x value
dim = 9
theta = [1e-2] * dim
tx = KRG(theta0=theta,print_prediction = False)
tx.set_training_values(x_train,y_train1)
tx.train()

# In[7]:
"""Predict x value and create plots"""

y1 = tx.predict_values(x_test)
print('kriging' + ' err: '+ str(compute_rms_error(tx,x_test,y_test1)))

fig, (ax1, ax2) = plt.subplots(2)

# first plot is to detect how far off the prediction is
ax1.plot(y_test1, y_test1, '-', label='$y_{true}$')
ax1.plot(y_test1, y1, 'r.', label='$\hat{y}$')
ax1.set_xlabel('$y_{true}$')
ax1.set_ylabel('response')
ax1.legend(loc='upper left')
ax1.set_title('KRG' + ' model: validation of the prediction model')  

# second plot is to plot the real data versus prediction
ax2.plot(x_test[:, 8], y1, 'b-', label = 'prediction of x_test')
ax2.plot(x_test[:, 8], y_test1, 'g--', label = 'true line')
ax2.set_xlabel('domain')
ax2.set_ylabel('response')
ax2.legend(loc = 'upper left')
plt.show()
print("theta values" + f'{tx.optimal_theta}')


# In[8]:
"""train model 2"""
ty = KPLS(theta0=theta,print_prediction = False, eval_n_comp = True)
ty.set_training_values(x_train,y_train2)
ty.train()

# In[9]:
"""Predict y value and create plots"""
y2 = ty.predict_values(x_test)
print('kriging' + ' err: '+ str(compute_rms_error(ty,x_test,y_test2)))

fig, (ax1, ax2) = plt.subplots(2)
# first plot is to detect how far off the prediction is
ax1.plot(y_test2, y_test2, '-', label='$y_{true}$')
ax1.plot(y_test2, y2, 'r.', label='$\hat{y}$')
ax1.set_xlabel('$y_{true}$')
ax1.set_ylabel('response')
ax1.legend(loc='upper left')
ax1.set_title('KRG' + ' model: validation of the prediction model')  

# second plot is to plot the real data versus prediction
ax2.plot(x_test[:, 8], y2, 'b-', label = 'prediction of y_test')
ax2.plot(x_test[:, 8], y_test2, 'g--', label = 'true line')
ax2.set_xlabel('domain')
ax2.set_ylabel('response')
ax2.legend(loc = 'upper left')
plt.show()
print("theta values" + f'{ty.optimal_theta}')


# # In[8]:
# #predict a path from the mechanisms the model has never seen (from the validation split) and compare to raw and prepro
# for i in range(10):
#     index = random.randint(0,len(x_validation)) #random sample from the validation data
#     #index = 2
# #for index in [1,2,3,4,5]:
#     path_pics=r"C:\LokaleDaten\s8913814\Documents\Python_experimental\220303_pics_coords"
    
#     coords_prepro_single = y_validation[index]
#     #coords_ML_single     = model.predict([x_validation[index]]).tolist()[0]        #@Daniel: put the predictions of your metamodell here
#     coords_RAW_single    = coords_RAW[coords_prepro.index(coords_prepro_single)]
    
#     compare_RAW_preprocessed_coords_in_plot(path                 = path_pics,
#                                             coords_RAW_single    = coords_RAW_single,
#                                             coords_prepro_single = coords_prepro_single,
#                                             #coords_ML_single     = coords_ML_single,
#                                             title                = str(index))
