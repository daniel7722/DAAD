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

# This class do from data preprocessing to building models for each position
class surrogate_modelling:

    def __init__(self, input_set = list_chromosome, output_set = coords_prepro):
        # initialise datasets
        self.input_set = input_set
        self.output_set = output_set

    def portion_data(self, test_size = 100, train_size = 150):
        # Taking a portion of data becasue too many data entries crash jupyter kernel (The virtual machine runs pretty slow on large dataset)
        self.testing_in = np.array(self.input_set[:test_size])
        self.testing_out = np.array(self.output_set[:test_size])
        self.training_in = np.array(self.input_set[test_size:train_size])
        self.training_out = np.array(self.output_set[test_size:train_size])

    def preprocessing(self, num_point):
        # sepatate output into x set and y set
        transpose_testing = np.array(self.testing_out).T
        self.num_point = num_point
        data_x_test = transpose_testing[:self.num_point]
        data_y_test = transpose_testing[self.num_point:]
        transpose_training = np.array(self.training_out).T
        data_x_train = transpose_training[:self.num_point]
        data_y_train = transpose_training[self.num_point:]

        # recreating the whole DataFrame with input and output and an index column 
        # testinig data 
        self.testing_processed     = []
        for index, i in enumerate(self.testing_in):
            for indexs, (j, z) in enumerate(zip(data_x_test, data_y_test)):
                _temp = []
                _temp.extend(i)
                _temp.extend([indexs])
                _temp.extend([j[index]])
                _temp.extend([z[index]])
                self.testing_processed.append(_temp)

        # training data
        self.training_processed    = []
        for index, i in enumerate(self.training_in):
            for indexs, (j, z) in enumerate(zip(data_x_train, data_y_train)):
                _temp = []
                _temp.extend(i)
                _temp.extend([indexs])
                _temp.extend([j[index]])
                _temp.extend([z[index]])
                self.training_processed.append(_temp)
        
        # making DataFrames for each set
        columns =list('0123456789')+['10']
        df_process_test = pd.DataFrame(self.testing_processed, columns= columns)
        df_process_train = pd.DataFrame(self.training_processed, columns= columns)

        # group the data by index I put previously so it would be like all zeros as the first configuration and all ones as the successive configuration
        self.df_process_train_split = df_process_train.groupby(by = ['8'])
        self.df_process_test_split = df_process_test.groupby(by = ['8'])
        # full1 = []
        # full2 = []
        # for j in range(10):
        #     daniel = []
        #     daniel1 = []
        #     for i in range(10):
        #         temp1 = self.df_process_test_split.get_group(i).reset_index()
        #         daniel.append(temp1.iloc[j, 10])
        #         temp2 = self.df_process_test_split.get_group(i).reset_index()
        #         daniel1.append(temp2.iloc[j, 11])
        #     full1.append(daniel)
        #     full2.append(daniel1)
        # return self.df_process_test_split.get_group(9).reset_index().iloc[:, 11]
        # # plt.plot(full1[9], full2[9], '.')
        # # self.df_process_test_split.get_group(5).reset_index()



    def slicing_training(self, rand_num):
        # Building model for each index group
        # initialise variables and provisonal list
        dim = 9
        theta = [1e-2] * dim
        x_value = []
        x_validate = []
        y_value = []
        y_validate = []
        
        for i in range(self.num_point):
            # separating it into input (x) and two outputs (y1, y2)
            self.x_test = self.df_process_test_split.get_group(i).reset_index().iloc[:, :9].to_numpy()
            self.y_test1 = self.df_process_test_split.get_group(i).reset_index().iloc[:, 10].to_numpy()
            self.y_test2 = self.df_process_test_split.get_group(i).reset_index().iloc[:, 11].to_numpy()

            # generating random number to select a set of training data from each index group
            indicesss = np.random.choice(range(len(self.df_process_train_split.get_group(0))), rand_num)

            # separating it into input (x1) and two outputs (y1, y2)
            self.x_train = self.df_process_train_split.get_group(i).reset_index().iloc[indicesss, :9].to_numpy()
            self.y_train1 = self.df_process_train_split.get_group(i).reset_index().iloc[indicesss, 10].to_numpy()
            self.y_train2 = self.df_process_train_split.get_group(i).reset_index().iloc[indicesss, 11].to_numpy()

            # training Kriging model for each index group on x
            tx = KRG(theta0=theta,print_prediction = False)
            tx.set_training_values(self.x_train,self.y_train1)
            tx.train()
            y1 = tx.predict_values(self.x_test)
            # saving predicted data into another list for plotting
            x_value.append(y1.ravel())
            x_validate.append(self.y_test1.ravel())
            
            # training KPLS model for each index group on y
            ty = KPLS(theta0=theta,print_prediction = False, eval_n_comp = True)
            ty.set_training_values(self.x_train, self.y_train2)
            ty.train()
            y2 = ty.predict_values(self.x_test)
            # saving predicted data into another list for plotting 
            y_value.append(y2.ravel())
            y_validate.append(self.y_test2.ravel())
        
        # make the data saved into datafame
        df_x = pd.DataFrame(x_value)
        df_y = pd.DataFrame(y_value)
        df_x_val = pd.DataFrame(x_validate)
        df_y_val = pd.DataFrame(y_validate)

        # plot out the index zeroth group with green dot plot as validation
        plt.plot(df_x.iloc[:, 0], df_y.iloc[:, 0], 'b.')
        plt.plot(df_x_val.iloc[:, 0], df_y_val.iloc[:, 0], 'g.')

# In[35]:
# calling class
foo = surrogate_modelling()
foo.portion_data(test_size = 100, train_size = 150)
foo.preprocessing(num_point = number_of_points)
foo.slicing_training(rand_num = 20)