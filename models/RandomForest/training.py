# forecast monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

demo = True # If true, the randomForest will be trained only on a part of data (until 2023-10-01)
dataset_path=f'models/RandomForest/data/bulo_burti_001.csv'
lag_day_to_remove=['lag1']


series = read_csv(dataset_path, header=0)
		
print(f"{series.shape}")
series = series.dropna()
print(f"{series.shape}")
series = series.rename(columns={'ds' : 'date'})
series = series.rename(columns={'y' : 'reading'})
series['date'] = pd.to_datetime(series['date'])
# Extract features from the date
series['year'] = series['date'].dt.year
series['month'] = series['date'].dt.month
series['day'] = series['date'].dt.day
series = series.drop(columns=['belet_weyne_level__m'])
for lag in lag_day_to_remove:
    series = series.loc[:, ~series.columns.str.contains(lag)]


series.head()
print(f"{series.shape}")

# Custom function to split data based on the year
def decided_data_split(data, target_name_column):
    start_test_date = '2023-10-01'
    test_mask = data['date'] >= start_test_date
    data = data.drop(columns=['date'])
    
    # Split the data into training and testing sets based on the mask
    train_data = data[~test_mask]
    test_data = data[test_mask]
    
    # Separate features and target
    trainX = train_data.drop(columns=[target_name_column])
    trainy = train_data[target_name_column]
    testX = test_data.drop(columns=[target_name_column])
    testy = test_data[target_name_column]
    
    return trainX, testX, trainy, testy

def based_decided_data_split(model, data):
    trainX, testX, trainy, testy = decided_data_split(data, 'reading')
    # Fit the model
    model.fit(trainX, trainy)


def train_model(model,series):
    trainX = series.drop(columns=['reading'])
    trainy = series['reading'] 
    # Fit the model
    model.fit(trainX, trainy)


from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
model = GradientBoostingRegressor(n_estimators=1000)

if demo:
    based_decided_data_split(model, series)
else:
    train_model(model,series)
joblib.dump(model, "models/RandomForest/random_forest.joblib")

