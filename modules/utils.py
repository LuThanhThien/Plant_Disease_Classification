import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pickle
import seaborn as sns
import re
import matplotlib.pyplot as plt
import time

def dataset_partition(dataset, train_split=0.8, val_split=0.1, test_plit=0.1, shuffle=True, shuffle_size=10000):
    data_size = len(dataset)

    # Shuffle if True
    if shuffle:
        dataset.shuffle(shuffle_size, seed=0)
    
    # Train and Validation size
    train_size = int(train_split*data_size)
    val_size = int(val_split*data_size)
    
    # Take dataset
    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size).skip(val_size)

    return train, val, test

# Save model with updated version
def save_model(model, folder_name):
    model_ver = max([int(i) for i in os.listdir(f"models/{folder_name}")]+[0]) + 1
    model.save(f"models/{folder_name}/{model_ver}")
    return None

# Load saved model
def load_model(model_name="latest", folder_name=None):
    if folder_name==None:
        raise NameError("Folder name to saved models must be specified")
    if model_name=='latest':
        model_ver = max([int(i) for i in os.listdir(f"models/{folder_name}")]+[0])
    else:
        try:
            model_ver = int(model_name) if int(model_name) > 0 else NameError("Model name must be a positive number")
        except:
            raise NameError("Model name must be a positive number")
    
    model = tf.keras.models.load_model(f"models/{folder_name}/{model_ver}")
    print(f'Successfully load model: {folder_name}/{model_ver}')
    
    return model

# Predict give label and confidence
def predict(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img.numpy())
    img_array = tf.expand_dims(img_array, 0) #Create a batch

    predictions = model.predict(img_array)

    predict_label = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)

    return predict_label, confidence

# precision, recall, f1-score and confusion metric
from sklearn.metrics import confusion_matrix, precision_score, recall_score
def precision_recall(model, test):
    # Extract labels from the test dataset
    test_labels = []
    for _, labels in test:
        test_labels.extend(labels.numpy())

    # Calculate the number of classes
    num_classes = len(set(test_labels))

    # Initialize lists to store per-class precision and recall
    precision_scores = []
    recall_scores = []
    confusion_matrix_values = np.zeros((num_classes, num_classes), dtype=int)

    for images, labels in test:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Calculate precision and recall for each class
        precision = precision_score(labels, predicted_labels, average=None)
        recall = recall_score(labels, predicted_labels, average=None)
        
        # Update confusion matrix
        confusion_matrix_values += confusion_matrix(labels, predicted_labels, labels=range(num_classes))
        
        # Append to the lists
        precision_scores.extend(precision)
        recall_scores.extend(recall)

    # Calculate average precision and recall across all classes
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    
    f1_score = 2 * (precision * recall) / (precision + recall)

    average_f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall)

    # Print average precision and recall
    print("Average Precision:")
    print(average_precision)
    print("Average Recall:")
    print(average_recall)
    print('Average F1 Score:')
    print(average_f1_score)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix_values)

    return precision_scores, recall_scores, f1_score, confusion_matrix_values 
# Plot confusion matrix as a heatmap
def confusion_heatmap(confusion_matrix_values, class_names, save=None):
    lenth=6
    plt.figure(figsize=(lenth+len(class_names)-2, lenth+len(class_names)-2))
    sns.heatmap(confusion_matrix_values, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if save!=None:
        plt.savefig(save)
    plt.show()


def plot_roc_curve(model, testset):
    
    y_scores_list = [model.predict(images) for images, _ in testset]
    y_scores = np.vstack(y_scores_list)
    y_true = np.concatenate([labels.numpy() for _, labels in testset], axis=0)

    roc_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# Avg test time per image
def evaluate_time(model, test, is_return=False):
    total_time = 0
    num_samples = 0
    
    for images, _ in test:

        start = time.time()
        model.predict(images)
        end = time.time()

        total_time += (end-start)
        num_samples += images.shape[0]
    
    avg_time = total_time/num_samples

    if avg_time*1000<1000: 
        print('Testing time: ', avg_time*1000, 'ms/sample')
    else: 
        print('Testing time: ', avg_time, 's/sample')
    
    if is_return: return avg_time
    return None
