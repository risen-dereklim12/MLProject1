# main_pipeline.py

from data_ingestion import load_data
from data_preprocessing import assign_ticket_type, preprocess_data
from model_training import train_and_test_model
from model_evaluation import evaluate_model
import numpy as np
import pandas as pd

# Load pre and post data
data_pre = load_data("data/cruise_pre.db", "cruise_pre")  
data_post = load_data("data/cruise_post.db", "cruise_post")

# Merge the 2 data
merged = data_pre.merge(data_post, on='Ext_Intcode')

# Removing the rows where `Cruise Name` and `Ticket Type` are `None` in the same row
merged = merged.drop(merged[(merged['Cruise Name'].isnull()) & (merged['Ticket Type'].isnull())].index)

# Replace NaN values in WiFi, Dining and Entertainment columns with 0 and non NaN values with 1
merged['WiFi'] = np.where(pd.isna(merged['WiFi']), 0, 1)
merged['Dining'] = np.where(pd.isna(merged['WiFi']), 0, 1)
merged['Entertainment'] = np.where(pd.isna(merged['Entertainment']), 0, 1)

# Apply the function to rows where Ticket Type is None
mask = merged['Ticket Type'].isnull()
merged.loc[mask, 'Ticket Type'] = merged[mask].apply(lambda x: assign_ticket_type(x['WiFi'], x['Dining'], x['Entertainment']), axis=1)

# Preprocess and train model on merged data
X, y = merged.drop('Ticket Type', axis=1), merged['Ticket Type']
X_preprocessed = preprocess_data(merged)

# Assuming you have X_preprocessed and y
model, X_test, y_test = train_and_test_model(X_preprocessed, y)

accuracy = evaluate_model(model, X_test, y_test)
print(f"Model (Pre-data) Accuracy: {accuracy*100:.2f}%")
