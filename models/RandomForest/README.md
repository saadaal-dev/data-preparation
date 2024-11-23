# Context
The objective of this script is to forecast the river level for a general station, with the demo specifically conducted for the Bulo Burti station.

This forecasting method initially utilized RandomForest for its predictions. After extensive testing with various data sources and mathematical approaches, it was determined that the most accurate model was the GradientBoostingRegressor. This model effectively predicts the river level for the following day.

Sure, here is a structured description of your dataset:

# Data

## Rain DATA
This section includes columns related to rainfall measurements. These columns capture various aspects of precipitation, such as the total amount of rainfall and the number of hours it rained. The data is further divided into different locations and lagged values, indicating the rainfall data from previous days. For example, columns like `ethiopia_fafen_haren_precipitation_sum` and `ethiopia_fafen_haren_precipitation_hours` provide the total rainfall and hours of rain for a specific location, while `ethiopia_fafen_haren_lag1_precipitation_sum` and `ethiopia_fafen_haren_lag1_precipitation_hours` represent the same data but for one day ago. Similar patterns are followed for other locations and lag periods.

## River level data
This section contains columns related to river levels. It includes both current and historical measurements of river levels at various locations. For instance, `lag1_level__m`, `lag3_level__m`, `lag7_level__m`, and `lag14_level__m` represent the river levels from 1, 3, 7, and 14 days ago, respectively. Additionally, there are columns for specific locations like `belet_weyne_level__m` and its lagged values such as `belet_weyne_lag1_level__m`, `belet_weyne_lag3_level__m`, etc. The target variable `y` represents the level of the Bulo Burti river, which is the main focus of the dataset. The column `ds` indicates the date of the measurements.

# Training
The prerequisite is to have the libraries in the `requirements.txt` installed. You can install them using the following command:
```bash
pip install -r requirements.txt
```
## Method
This model uses two main techniques to model the river level behavior:

- **RandomForest**: Initially, we considered using RandomForest for its ability to capture complex relationships between features. However, we found that the **GradientBoostingRegressor** was more accurate for our case. This method builds an additive model in a forward stage-wise fashion, optimizing for the loss function. It combines the predictions of several base estimators to improve robustness and accuracy. This technique is particularly effective in handling non-linear relationships and interactions between the features, such as rainfall and river levels.

It is important to note that this model can be enriched and fine-tuned. However, it is usable as it is.
Sure, here is the revised paragraph with an easy explanation of the variables:

## Variables to Change
- `demo = True`: If set to `True`, the RandomForest model will be trained only on a part of the data (up until 2023-10-01). This is useful for testing purposes (what we've done in the demo). If is not a demo put is as False, in this way the model will train on the whole dataset and can have better prediction.
- `dataset_path = 'xxxx/bulo_burti_001.csv'`: This is the path to the dataset file. Make sure to use the correct path.
- `lag_day_to_remove = ['lag1']`: This specifies the lag days to be removed from the equation. For example, `['lag1']` means the data from 1 day ago will be excluded from the prediction.

## Run
To run the training, please ensure you have the correct CSV files in the `data` folder. Place yourself at the same level as this README and run:
```bash
python3 model_training.py
```

# Inference

Before running the inference, make sure to change the same variables as before: `demo`, `dataset_path`, and `lag_day_to_remove`. 
If is not a Demo, put `demo=False` and utilize the correct dataset_path.
The prerequisite is to have the libraries in the `requirements.txt` installed. Place yourself at the same level as this README and run:
```bash
python3 model_inference.py
```
