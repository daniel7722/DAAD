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
# /Users/danielhuang/coding/DAAD_main/Data/220602_RAW_data_population_WITHOUT_COMMENT
# C:\Users\Administrator\Documents\Neuer Ordner\DAAD\Data\220602_RAW_data_population_WITHOUT_COMMENT
path_data=r'/Users/danielhuang/coding/DAAD_main/Data/220602_RAW_data_population_WITHOUT_COMMENT'

list_chromosome, list_fitness, coords_RAW = read_RAW_data(path=path_data)

#create the distributed points
number_of_points = 50 #too little and you lose information about the fea path
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



# In[3]:

class surrogate_modelling:

    def __init__(self, input_set = coords_prepro, output_set = list_chromosome):
        # initialise datasets
        self.input_set = input_set
        self.output_set = output_set

    def portion_data(self, test_size = 100, train_size = 150):
        # Taking a portion of data becasue too many data entries crash jupyter kernel
        self.testing_in = np.array(self.input_set[:test_size])
        self.testing_out = np.array(self.output_set[:test_size])
        self.training_in = np.array(self.input_set[test_size:train_size])
        self.training_out = np.array(self.output_set[test_size:train_size])

    def preprocessing(self):
        # sepatate output into x set and y set
        transpose_testing = np.array(self.testing_in).T
        data_x_test = transpose_testing[:50]
        data_y_test = transpose_testing[50:]
        transpose_training = np.array(self.training_in).T
        data_x_train = transpose_training[:50]
        data_y_train = transpose_training[50:]

        # cleaning and recreating data that is usable as I wish
        # testing data
        self.testing_processed     = []
        for index, i in enumerate(self.testing_out):
            for indexs, (j, z) in enumerate(zip(data_x_test, data_y_test)):
                _temp = []
                _temp.extend(i)
                _temp.extend([indexs])
                _temp.extend([j[index]])
                _temp.extend([z[index]])
                self.testing_processed.append(_temp)

        # training data
        self.training_processed    = []
        for index, i in enumerate(self.training_out):
            for indexs, (j, z) in enumerate(zip(data_x_train, data_y_train)):
                _temp = []
                _temp.extend(i)
                _temp.extend([indexs])
                _temp.extend([j[index]])
                _temp.extend([z[index]])
                self.training_processed.append(_temp)
        
        columns =list('0123456789')+['10']
        df_process_test = pd.DataFrame(self.testing_processed, columns= columns)
        df_process_train = pd.DataFrame(self.training_processed, columns= columns)

        self.df_process_train_split = df_process_train.groupby(by = ['8'])
        self.df_process_test_split = df_process_test.groupby(by = ['8'])
        full1 = []
        full2 = []
        for j in range(10):
            daniel = []
            daniel1 = []
            for i in range(10):
                daniel.append(self.df_process_test_split.get_group(i).iloc[j, 9])
                daniel1.append(self.df_process_test_split.get_group(i).iloc[j, 10])
            full1.append(daniel)
            full2.append(daniel1)
        plt.plot(full1[1], full2[1], '.')



    def slicing_training(self, rand_num):
        dim = 8
        theta = [1e-2] * dim
        pre = []
        x_value = []
        x_validate = []
        y_value = []
        y_validate = []
            
        for i in range(10):
            # separating it into input and two outputs
            self.x_test = self.df_process_test_split.get_group(i).iloc[:, :8].to_numpy()
            self.y_test1 = self.df_process_test_split.get_group(i).iloc[:, 9].to_numpy()
            self.y_test2 = self.df_process_test_split.get_group(i).iloc[:, 10].to_numpy()

            indicesss = np.random.choice(range(len(self.df_process_train_split.get_group(0))), rand_num)

            # separating it into input and two outputs
            self.x_train = self.df_process_train_split.get_group(i).iloc[indicesss, :8].to_numpy()
            self.y_train1 = self.df_process_train_split.get_group(i).iloc[indicesss, 9].to_numpy()
            self.y_train2 = self.df_process_train_split.get_group(i).iloc[indicesss, 10].to_numpy()

            tx = KRG(theta0=theta,print_prediction = False)
            tx.set_training_values(self.x_train,self.y_train1)
            tx.train()
            y1 = tx.predict_values(self.x_test)
            x_value.append(y1.ravel())
            x_validate.append(self.y_test1.ravel())
            

            ty = KPLS(theta0=theta,print_prediction = False, eval_n_comp = True)
            ty.set_training_values(self.x_train, self.y_train2)
            ty.train()
            y2 = ty.predict_values(self.x_test)
            y_value.append(y2.ravel())
            y_validate.append(self.y_test2.ravel())
        
        df_x = pd.DataFrame(x_value)
        df_y = pd.DataFrame(y_value)
        df_x_val = pd.DataFrame(x_validate)
        df_y_val = pd.DataFrame(y_validate)

        plt.plot(df_x.iloc[0, :], df_y.iloc[0, :], 'b.')
        plt.plot(df_x_val.iloc[0, :], df_y_val.iloc[0, :], 'g.')

        


            # print('kriging' + ' err: '+ str(compute_rms_error(tx,self.x_test,self.y_test1)))

            # fig, (ax1, ax2) = plt.subplots(2)

            # # first plot is to detect how far off the prediction is
            # ax1.plot(self.y_test1, self.y_test1, '-', label='$y_{true}$')
            # ax1.plot(self.y_test1, y1, 'r.', label='$\hat{y}$')
            # ax1.set_xlabel('$y_{true}$')
            # ax1.set_ylabel('response')
            # ax1.legend(loc='upper left')
            # ax1.set_title('KRG' + ' model: validation of the prediction model')  

            # # second plot is to plot the real data versus prediction
            # ax2.plot([oo for oo in range(50)], y1, 'b-', label = 'prediction of x_test')
            # ax2.plot([pp for pp in range(50)], self.y_test1, 'g--', label = 'true line')
            # ax2.set_xlabel('domain')
            # ax2.set_ylabel('response')
            # ax2.legend(loc = 'upper left')
            # plt.savefig(f'Pics/position{i}.png')
            # print("theta values" + f'{tx.optimal_theta}')
            


# In[35]:
foo = surrogate_modelling(input_set = coords_prepro, output_set = list_chromosome)
foo.portion_data(test_size = 100, train_size = 150)
foo.preprocessing()
# foo.slicing_training(rand_num = 20)


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

# In[4]:
"""Data Splitting"""

# Taking a portion of data becasue too many data entries crash jupyter kernel
testing_data1 = np.array(coords_prepro[:100])
testing_data2 = np.array(list_chromosome[:100])
training_data1 = np.array(coords_prepro[100:150])
training_data2 = np.array(list_chromosome[100:150])

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

