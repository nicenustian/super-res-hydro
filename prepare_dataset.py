import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
import h5py


def prepare_dataset(dataset_dir, dataset_file_filter, quantities, 
                    batch_size, examples = None):
       
    file_list = os.listdir(dataset_dir)
    filtered_files = [filename for filename in file_list
                              if (dataset_file_filter in filename)]
            
    file_name = sorted(filtered_files)
    
    print('Files in the dataset folder, starting from lowest to highest resolution..')
    print(file_name)
        
    data = []
    
    # Calculate the number of examples that are multiples of the batch size
    examples = (examples // batch_size) * batch_size
        
    for name in file_name:
        file_extension = os.path.splitext(name)[1]

        if file_extension == ".npy":
            # Load the data from the .npy file
            npz_file = np.load(dataset_dir+name, allow_pickle=True)

            # Extract the arrays from the keys_list
            file_data = np.stack([npz_file[key][:examples] for key in quantities], axis=-1)
        
        elif file_extension == ".hdf5":
            # Load the data from the .h5 file
                        
            with h5py.File(dataset_dir+name, 'r') as f:
                file_data = f['data'][:]
            file_data = file_data[:examples, ..., :len(quantities)]
            
        else:
            raise ValueError("Unsupported file format. Only .npy and .hdf5 files are supported.")


        data.append(file_data)


    stacked_data = np.vstack(data)
    
    num_features = stacked_data.shape[-1]
    original_dim = stacked_data.shape[1]
        
    #standard scaler do not works
    scaler = MinMaxScaler()
    ###scaler = StandardScaler()
    data_shape = stacked_data.shape
    stacked_data = scaler.fit_transform(stacked_data.reshape(-1, num_features)).reshape(data_shape)
        
    for di in range(len(data)):
        
        print("file index", di+1, data[di].shape)
        
        for ki in range(len(quantities)):
                     
            min_value = np.min(data[di][...,ki])
            max_value = np.max(data[di][...,ki])
            mean_value = np.mean(data[di][...,ki])
            std_value = np.std(data[di][...,ki])
         
            print(quantities[ki], "Min:", "{:.4f}".format(min_value),
                  "Max:", "{:.4f}".format(max_value),
                  "Mean:", "{:.4f}".format(mean_value),
                  "Std:", "{:.4f}".format(std_value))
         
        data_shape = data[di].shape
        data[di] = scaler.transform(data[di].reshape(-1, num_features)).reshape(data_shape)
    
    print('After preprocessing..')
    
    for fi in range(len(data)):
        print("file index", fi+1, data_shape)
        for ki in range(len(quantities)):
        
            min_value = np.min(data[fi][...,ki])
            max_value = np.max(data[fi][...,ki])
            mean_value = np.mean(data[fi][...,ki])
            std_value = np.std(data[fi][...,ki])
         
            print(quantities[ki], "Min:", "{:.4f}".format(min_value),
                  "Max:", "{:.4f}".format(max_value),
                  "Mean:", "{:.4f}".format(mean_value),
                  "Std:", "{:.4f}".format(std_value))
    
    print('datasets', len(data))

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
               tf.data.experimental.AutoShardPolicy.FILE
    
    # the only way it can batch mutiple list of datasets    
    # Create datasets from tensors
    datasets = [tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(array, dtype=tf.float32)) for array in data]

    # Zip datasets together to batch them
    datasets = tf.data.Dataset.zip(tuple(datasets))    
    datasets = datasets.with_options(options)


    return data, datasets, examples, original_dim, num_features
