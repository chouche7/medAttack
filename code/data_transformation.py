#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:56:45 2023

@author: chouche7
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle

max_sequence_length = 48  # Define the maximum sequence length (48 hours in this case)

def feature_normalization(feature_ids, data):
    for f_id in feature_ids:
        mask = data['ITEMID'] == f_id
        item_data = data[mask]["NORM_VALUENUM"]
        
        max_val = item_data.abs().max()
        min_val = item_data.abs().min()

        item_data = (item_data - min_val)/(max_val - min_val)
        data.loc[mask, "NORM_VALUENUM"] = item_data
        
    return data

def remove_outliers(feature_ids, data):
    
    for f_id in feature_ids:
        mask = data['ITEMID'] == f_id
        item_data = data[mask]

        # Impute missing values for each ITEMID using mean across time stamps
        imputed_data = item_data.groupby('SUBJECT_ID')["VALUENUM"].transform(lambda x: x.fillna(x.mean()))

        # Flatten the DataFrame for outlier detection
        flattened_data = imputed_data.values.flatten()
    
        # Calculate IQR values for each feature
        Q1 = np.percentile(flattened_data, 25)
        Q3 = np.percentile(flattened_data, 75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify and remove outliers
        outliers = (flattened_data < lower_bound) | (flattened_data > upper_bound)
        outliers_indices = np.where(outliers)[0]
    
        for idx in outliers_indices:
            if idx > 0:
                flattened_data[idx] = flattened_data[idx - 1]  # Impute outlier using carry-forward method
            else:
                flattened_data[idx] = flattened_data.mean()  # Impute outlier using overall mean if it's the first entry
    
        # Reshape the flattened data back to the original DataFrame shape
        #imputed_data[:] = flattened_data.reshape(imputed_data.shape)
        
        # Update the original DataFrame with the imputed values
        data.loc[mask, "VALUENUM"] = flattened_data

    return data


# Function to process and pad/truncate sequences
def process_sequences(data, feature_ids, n_train, n_features):
    
    processed_data = np.zeros((n_train, max_sequence_length, n_features))

    i = 0
    mask_value = 0
    
    # Pad short sequences and truncate long sequences
    for key, df in data:
        #transform dataframe into DATETIME x features
        transformed = df.groupby(["CHARTTIME", "ITEMID"])["NORM_VALUENUM"].aggregate("mean").unstack()
        
        #fill in missing features
        missing = n_features - transformed.shape[1]
        missing_ids = np.setdiff1d(feature_ids, np.array(transformed.columns))
        missing_features = np.zeros((len(transformed), missing))
        missing_features = pd.DataFrame(missing_features, columns = missing_ids)
        transformed = pd.concat((transformed, missing_features), axis = 1)
        transformed = transformed.reindex(sorted(transformed.columns), axis=1)
        
        if len(transformed) >= max_sequence_length:
            processed_data[i] = transformed[-max_sequence_length:]  # Truncate long sequences
        else:
            pad_length = max_sequence_length - len(transformed)
            padding = np.full((pad_length, n_features), mask_value)  # Define the masking value
            processed_data[i] = np.vstack((padding, transformed))  # Pad short sequences
        
        i+=1
    return processed_data

mortality = pd.read_csv("../../mimic3/data/MORTALITY.csv", index_col=0)
lab_events = pd.read_csv('../../mimic3/data/LABEVENTS_SHORT.csv', index_col=0)
vital_signs = pd.read_csv("../../mimic3/data/CHARTEVENTS_SHORT.csv", index_col=0)

#lab_items = pd.read_csv("../../mimic3/data/D_LABITEMS.csv")
#vital_items = pd.read_csv("../../mimic3/data/D_ITEMS.csv")

#merge all the data into one big dataframe
features_df = pd.concat((lab_events, vital_signs))
features_df = features_df.merge(mortality, on = "SUBJECT_ID", how="left")
feature_ids = features_df['ITEMID'].unique()

#Sort Data by Timestamps:
features_df.sort_values(["SUBJECT_ID", "CHARTTIME"], inplace = True)

#IMPUTATION & OUTLIER REMOVAL
features_df = remove_outliers(feature_ids, features_df)

#NORMALIZATION
features_df["NORM_VALUENUM"] = features_df["VALUENUM"]
features_df = feature_normalization(feature_ids, features_df)
features_df = features_df.drop(columns = "VALUENUM")
label = features_df[["SUBJECT_ID", "mortality"]].drop_duplicates()
label = label.reset_index(drop=True)

# Group data by SUBJECT_ID
grouped = features_df.groupby(['SUBJECT_ID'])


# Initialize a 3D array to store the data and padd sequences
n_train = len(features_df['SUBJECT_ID'].unique())
n_features = 19
feature_3d = process_sequences(grouped, feature_ids, n_train, n_features)
feature_3d = np.nan_to_num(feature_3d)

X = feature_3d
y = np.array(label["mortality"])
#ids = np.array(label["SUBJECT_ID"])

# Stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

np.random.seed(0)

fold = 0
    
for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #ids_train, ids_test = ids[train_index], ids[test_index]
    
    # Positive and negative class indices
    positive_indices = np.where(y_train == 1)[0]
    negative_indices = np.where(y_train == 0)[0]

    # Down-sample negative class to balance the training data
    # we select the same number of negative class as positive class
    num_pos = len(positive_indices)
    num_neg = len(negative_indices)
    downsampled_indices = np.random.choice(negative_indices, num_pos, replace=False)

    # Combine down-sampled negative class with the positive class
    balanced_indices = np.concatenate((positive_indices, downsampled_indices))

    X_balanced = X_train[balanced_indices]
    y_balanced = y_train[balanced_indices]
    #ids_balanced = ids_train[balanced_indices]
    
    # Split the balanced training data into 4:1 ratio for training and validation set
    X_train_bal, X_val, y_train_bal, y_val, train_ind, test_ind = train_test_split(X_balanced, y_balanced, np.arange(len(y_balanced)), test_size=0.2, random_state=0)
    
    #ids_train_bal = ids_balanced[train_ind]
    #ids_test_bal = ids_balanced[test_ind]
    
    #y_train_bal = pd.DataFrame(np.vstack((ids_train_bal, y_train_bal)).T)
    #y_val = pd.DataFrame(np.vstack((ids_test_bal, y_val)).T)
    
    train_target = np.zeros((len(y_train_bal), 2))
    train_target[np.arange(len(y_train_bal)), y_train_bal] = 1
    
    test_target = np.zeros((len(y_val), 2))
    test_target[np.arange(len(y_val)), y_val] = 1
    
    
    train_X_path = '../../mimic3/fold_' + str(fold) + "/data_train.pkl"
    test_X_path = '../../mimic3/fold_' + str(fold) + "/data_test.pkl"
    train_Y_path = '../../mimic3/fold_' + str(fold) + "/target_train.pkl"
    test_Y_path = '../../mimic3/fold_' + str(fold) + "/target_test.pkl"
    
    # Save as Pickle files
    with open(train_X_path, 'wb') as f:
        pickle.dump(X_train_bal, f)
    
    with open(test_X_path, 'wb') as f:
        pickle.dump(X_val, f)
        
    with open(train_Y_path, 'wb') as f:
        pickle.dump(train_target, f)
        
    with open(test_Y_path, 'wb') as f:
        pickle.dump(test_target, f)
    
    fold+=1