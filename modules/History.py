import matplotlib.pyplot as plt
import numpy
import os
import pickle
import keras

def plot_loss_metric(hist):
    # Plot performance of model
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(1,2,1)
    hist_type = type(hist)
    if hist_type!=dict:
        plt.plot(hist.history['loss'], color='orange', label='loss')
        plt.plot(hist.history['val_loss'], color='teal', label='val loss')

        ax = plt.subplot(1,2,2)
        plt.plot(hist.history['accuracy'], color='orange', label='accuracy')
        plt.plot(hist.history['val_accuracy'], color='teal', label='val accuracy')

    else:

        plt.plot(hist['loss'], color='orange', label='loss')
        plt.plot(hist['val_loss'], color='teal', label='val loss')

        ax = plt.subplot(1,2,2)
        plt.plot(hist['accuracy'], color='orange', label='accuracy')
        plt.plot(hist['val_accuracy'], color='teal', label='val accuracy')


    return None

#Marge history
def merge_hist(hist_list):
    # Extract metrics from the History objects
    metrics_to_concat = ['loss', 'accuracy', 'val_loss', 'val_accuracy']  # Add other metrics if needed
    merged_metrics = {metric: [] for metric in metrics_to_concat}
    for metric in metrics_to_concat:
        for hist in hist_list:
            if type(hist) == keras.callbacks.History:
                merged_metrics[metric] += hist.history[metric]
            elif type(hist) == dict:
                merged_metrics[metric] += hist[metric]
            else:
                raise TypeError("Only History or Dict are allowed")
                return
    print('Merged successfully!')

    return merged_metrics

# Save history
def save_hist(hist, folder_name):

    saved_path = f"history\{folder_name}"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path, exist_ok=False)
    
    # Define the full path to the file
    pkl_files = [filename for filename in os.listdir(f"history\{folder_name}") if filename.endswith('.pkl')]+['0.pkl']
    hist_ver = max(int(filename.split('.')[0]) for filename in pkl_files) + 1
    file_path = os.path.join(saved_path, f'{hist_ver}.pkl')

    # Save the metrics_to_concat dictionary to a file
    with open(file_path, 'wb') as f:
        pickle.dump(hist, f)
    print(f'Save successfully into {file_path}')
    return None
    
    
def load_hist(folder_name, hist_name='latest'):
    if hist_name=='latest':
        pkl_files = [filename for filename in os.listdir(f"history\{folder_name}") if filename.endswith('.pkl')]
        hist_name = max(int(filename.split('.')[0]) for filename in pkl_files)
    
    load_path = f"history\{folder_name}\{hist_name}.pkl"

    # Load the saved metrics_to_concat dictionary from the file
    with open(load_path, 'rb') as f:
        loaded_metrics = pickle.load(f)
    print(f'Successfully loaded: {load_path}')
    return loaded_metrics
