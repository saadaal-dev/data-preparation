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
from matplotlib.ticker import MaxNLocator

demo = False # If true, the randomForest will be trained only on a part of data (until 2023-10-01)
# If is a demo, this csv should also contains the y with the result data!
dataset_path=f'models/RandomForest/data/bulo_burti_001.csv'
lag_day_to_remove=['lag1']



series = read_csv(dataset_path, header=0)
		
print(f"{series.shape}")
series = series.dropna()
print(f"{series.shape}")
series = series.rename(columns={'ds' : 'date'})
if demo:
    series = series.rename(columns={'y' : 'reading'})
if 'y' in series.columns:
    series = series.drop(columns={'y'})

series['date'] = pd.to_datetime(series['date'])
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
    test_data = data[test_mask]
    
    # Separate features and target
    testX = test_data.drop(columns=[target_name_column])
    testy = test_data[target_name_column]
    
    return testX, testy

def based_decided_data_split(model, data):
    testX, testy = decided_data_split(data, 'reading')
    # Make a one-step prediction
    yhat = model.predict(testX)
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(testy, yhat)
    RMSE =((testy - yhat) ** 2).mean() ** .5

    print(f"MSE: {mse}  RMSE: {RMSE} decided_data_split")
    # plot expected vs predicted
    pyplot.title(f"{model} MSE: {mse}  RMSE: {RMSE} decided_data_split")
    pyplot.plot(testy.values, label='Expected')
    pyplot.plot(yhat, label='Predicted')
    pyplot.legend()
    pyplot.savefig(f"models/RandomForest/mseDemo.png")  # Save the plot to a file
    pyplot.close()  # Close the plot to free up memory


def train_model(model,series):
    trainX = series.drop(columns=['reading'])
    trainy = series['reading'] 
    # Fit the model
    model.fit(trainX, trainy)

model = joblib.load(f"models/RandomForest/random_forest.joblib")

if demo:
    based_decided_data_split(model , series)
else:
    # Save the columns
    x_ax = series['date'].tolist()
    series = series.drop(columns=['date'])
    yhat = model.predict(series)
    print(yhat)

    fig, ax = pyplot.subplots()
    ax.plot(x_ax, yhat, label='Predicted')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.legend()
    pyplot.savefig(f"models/RandomForest/predicted_plot.png")
    pyplot.close()