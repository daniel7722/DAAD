# Surrogate Model for Soft Robot Movement Prediction (DAAD)

## Project Description:

This project is about using surrogate modelling techniques to capture and predict the movement of a simulated soft robot in 2D. Surrogate model is a method to solve a computationally expensive blackbox problem by building a simplified, middle-fidelity, but quick model from the real situations. This allows quicker run time to reach the answer and helps do further actions more accountable. 

We started from data_creation.py. This is a file that we use our defined functions to generate non-linear data. And in surrogate_trial1.py, we try to use Kriging method and KPLS method to test on the feasibility of implementing surrogate model. 

Then, we deal with the real problem by generating data using _220302_data_creation_for_NN_ADDITION.py. And we train the model on surrogate_model.py.

Currently, I am facing kernel crashing problem when running data entries that is more than a certain extent. Also, the accuracy is not that promising. So, the future work might be choosing the suitable surrogate model method and hyperparameters tuning.

Surrogate modelling is relatively a new field of study and is certainly still developing. Therefore, there are still a lot to explore in this project and might have hidden potential that I haven't found yet. It's been a great journey to make it this far and I will update if there's anything I wanna try and actually works. Thank you