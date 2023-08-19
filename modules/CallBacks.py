import tensorflow as tf
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

def checkpoint_call(folder_name):
    model_ver = max([int(i) for i in os.listdir(f"checkpoints/{folder_name}")]+[0]) + 1
    filepath=f"checkpoints/{folder_name}/{model_ver}/"+"{epoch}.ckpt"
    return ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                           save_weights_only=True, save_best_only=True, mode='min')


# Load weights by lastest checkpoint
def load_latest_checkpoint(model, directory):
    model_ver = max([int(i) for i in os.listdir(f"checkpoints/{directory}")]+[0])
    latest = tf.train.latest_checkpoint(f"checkpoints/{directory}/{model_ver}")
    model.load_weights(latest)
    print('Loaded: ',latest)

    
# Load weights by lastest checkpoint
def load_checkpoint(model, directory, checkpoint_name=None):
    if checkpoint_name==None:
        model_ver = max([int(i) for i in os.listdir(f"checkpoints/{directory}")]+[0])
        loaded_checkpoint = tf.train.latest_checkpoint(f"checkpoints/{directory}/{model_ver}")
        raise Warning("""Latest checkpoint will be loaded. 
                      Please, mention checkpoint name in 'checkpoint_name' if you want to load a specific checkpoint""")
    else: 
        model_ver = checkpoint_name
        loaded_checkpoint = f"checkpoints/{directory}/{model_ver}.ckpt"
    model.load_weights(loaded_checkpoint)
    print('Loaded: ',loaded_checkpoint)