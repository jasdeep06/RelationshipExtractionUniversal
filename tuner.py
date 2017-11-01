import numpy as np
from train import train
from hyperparameters import hyperparameters
import random


def tune(n_times=25):
    for i in range(n_times):
        print("Running tuning iteration number {}".format(i))
        hyperparameter_setting=hyperparameters(batch_size=(10,100),num_units=(16,150),embedding_size=(40,100),learning_rate=(.000001,1))
        hyperparameter_setting.batch_size=sample_randomly(hyperparameter_setting.batch_size,use_log_scale=False,sample_int=True,from_list=False)
        hyperparameter_setting.num_units=sample_randomly(hyperparameter_setting.num_units,use_log_scale=False,sample_int=True,from_list=False)
        hyperparameter_setting.embedding_size=sample_randomly(hyperparameter_setting.embedding_size,use_log_scale=False,sample_int=True,from_list=False)
        hyperparameter_setting.learning_rate=sample_randomly(hyperparameter_setting.learning_rate,use_log_scale=True,sample_int=False,from_list=False)
        print("The configuration is ",hyperparameter_setting.__dict__)
        train(resume_training=False,total_epochs=5,test_after_n_epoch=2,disp_loss_after_n_iter=10,merge_summary_after_n_iter=10,hyperparameters=hyperparameter_setting)

def sample_randomly(range,use_log_scale=False,sample_int=False,from_list=False):
    if sample_int==True and use_log_scale==False:
        lower_val,upper_val=range
        sample=np.random.randint(low=lower_val,high=upper_val+1)
        return sample

    if sample_int==False and use_log_scale==True:
        lower_val,upper_val=range
        lower_val=np.log10(lower_val)
        upper_val=np.log10(upper_val)

        sample=random.uniform(lower_val,upper_val)
        sample=10**sample
        return sample

    if from_list==True:
        sample=np.random.randint(low=0,high=len(range))
        return range[sample]


if __name__ == '__main__':
    tune(25)
