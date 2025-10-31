import os.path
import numpy as np


feature = np.loadtxt(r'D:\ResearchDoc\MachineLearing\PycharmDoc\CB_inversedesign\Design_AFM_revision\all_data'
                     r'\data_abaqus_model_250626_biaxial\data_all_feature_uniaxial_y.txt')
rf_y = np.loadtxt(r'D:\ResearchDoc\MachineLearing\PycharmDoc\CB_inversedesign\Design_AFM_revision\all_data'
                  r'\data_abaqus_model_250626_biaxial\data_all_rfy_biaxial.txt')
dataset_y = rf_y
dataset_x = feature
num_all_samples = dataset_x.shape[0]
num_train_data = 9000
num_feature = dataset_x.shape[1]
dataset = np.concatenate((dataset_x, dataset_y), axis=1)
random_chosen_samples_index = np.random.choice(num_all_samples, size=num_train_data, replace=False)
other_samples_index = np.setdiff1d(np.arange(1, num_all_samples), random_chosen_samples_index)
train_data = dataset[random_chosen_samples_index, :]
test_data = dataset[other_samples_index, :num_feature]
test_data_value = dataset[other_samples_index, num_feature:]
print(train_data.shape)
print(test_data.shape)
print(test_data_value.shape)
save_path = './dataset'
np.savetxt(os.path.join(save_path, 'train_data.txt'), train_data)
np.savetxt(os.path.join(save_path, 'test_data.txt'), test_data)
np.savetxt(os.path.join(save_path, 'test_data_value.txt'), test_data_value)
