# data_preprocessing.py

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from datetime import datetime

# Define the function to determine the Ticket Type
def assign_ticket_type(wifi, dining, entertainment):
    if wifi:
        return "luxury"
    elif entertainment and not wifi:
        return "deluxe"
    elif dining and not wifi and not entertainment:
        return "standard"
    else:
        return None
    
# Convert 'Cruise Distance' values to numeric and remove 'KM' and 'Miles'
def standardize_distance(distance):
    if pd.isna(distance):
        return distance
    if 'KM' in distance:
        return float(distance.replace('KM', '').strip())
    elif 'Miles' in distance:  # Assuming that 1 Mile : 1.60934 KM
        return float(distance.replace('Miles', '').strip()) * 1.60934
    else:
        return float(distance)
    
def convert_age(data):
    # Convert 'Date of Birth' to datetime format
    data['Date of Birth'] = pd.to_datetime(data['Date of Birth'], errors='coerce', format='%d/%m/%Y')

    # Current year
    current_date = datetime.now()

    # Check if the birthday has not occurred yet for this year
    before_birthday = (data['Date of Birth'].dt.month > current_date.month) | ((data['Date of Birth'].dt.month == current_date.month) & (data['Date of Birth'].dt.day > current_date.day))
    data['Age'] = current_date.year - data['Date of Birth'].dt.year - before_birthday.astype(int)

    # Handle rows where 'Date of Birth' is NaT (i.e., not a time)
    data['Age'] = data['Age'].fillna(0).astype(int)

    # Replace age == 0 with median age
    median_age = data[data["Age"] > 0]["Age"].median()
    data.loc[data["Age"] == 0, "Age"] = median_age
    return data

def fillNa(data):
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    return data

def preprocess_data(data):

    # Replacing the Cruise Names that are similar but with discrepancies in lower case e.g. blast0ise to blastoise
    data['Cruise Name'] = data['Cruise Name'].str.replace(r'(?i)blast[o0]ise', 'blastoise', regex=True)
    data['Cruise Name'] = data['Cruise Name'].str.replace(r'(?i)^blast$', 'blastoise', regex=True)
    data['Cruise Name'] = data['Cruise Name'].str.replace(r'(?i)[lI]apras', 'lapras', regex=True)
    data['Cruise Name'] = data['Cruise Name'].str.replace(r'(?i)^lap$', 'lapras', regex=True)

    # Standardise the metrics of Cruise Distance
    data['Cruise Distance'] = data['Cruise Distance'].apply(standardize_distance)

    # Impute None values in 'Cruise Distance' with the median
    median_distance = data['Cruise Distance'].median()
    data['Cruise Distance'].fillna(median_distance, inplace=True)

    # There are None values in Gender. I will make half of it to be Male and half to Female to prevent 
    # distorting the data
    # Find the indices where Gender is None
    none_indices = data[data['Gender'].isnull()].index.tolist()
    # Calculate half length
    half_len = len(none_indices) // 2
    # Set first half to Male and second half to Female
    data.loc[none_indices[:half_len], 'Gender'] = 'Male'
    data.loc[none_indices[half_len:], 'Gender'] = 'Female'

    #Remove logging, ext_intcode, index_x and index_y
    data = data.drop(columns=['Ext_Intcode', 'index_x', 'index_y', 'Logging'])

    #Change dob to age
    data = convert_age(data)

    #Fill each column  with None or NaN values with the mode of column values
    data = fillNa(data)

    numeric_features = ['Embarkation/Disembarkation time convenient', 
                        'Ease of Online booking', 'Cruise Distance',
                        'Online Check-in', 'Cabin Comfort',
                        'Cabin service', 'Baggage handling',
                        'Port Check-in Service', 'Onboard Service', 'Cleanliness',
                        'Age'] 
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['Gender', 'Source of Traffic', 'Cruise Name', 'WiFi', 'Dining', 
                            'Entertainment', 'Onboard Wifi Service', 'Onboard Entertainment', 
                            'Onboard Dining Service']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    data_preprocessed = preprocessor.fit_transform(data)
    return data_preprocessed
