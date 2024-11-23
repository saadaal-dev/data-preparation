# ARIMA Model for River Level Forecasting

## Introduction

This approach implements an ARIMA (AutoRegressive Integrated Moving Average) model using the `sktime` library to forecast river levels based on precipitation data. The goal is to predict future river levels by analyzing historical precipitation and river level data.

## Project Structure

- **`arimasktime.py`**: Contains functions for data preparation, model training, forecasting, and plotting.
- **`sktime_notebook.ipynb`**: Jupyter notebook for interactive forecasts and visualization.
- **Data Files**: CSV files containing historical river level and weather data.

## Data Sources

- **River Level Data**: `bulo_burti_river_station.csv`
- **Weather Data**: `Ethiopia_Gabredarre_Fafen_river_midpoint_historical_weather_daily_2024-11-19.csv`

## Prerequisites

- Python 3.10 or higher
- Recommended: Use a virtual environment

## Installation

1. **Clone the repository**
2. Navigate to the project directory
3. Create and activate a virtual environment
4. Install required packages

## Usage

### Running the Script
- Execute the main script to train the model, forecast river levels, and plot the results.

### Using the Jupyter Notebook
- Run the Jupyter notebook for an interactive experience.

### Configuration

- **Forecast Dates**:  
  Modify the `forecast_start_date` and `forecast_end_date` variables in `arimasktime.py` or `sktime_notebook.ipynb` to change the forecast period.
  
- **Model Filename**:  
  Change the `model_filename` variable to save or load different models.

- **Data File Paths**:  
  Update the `BULO_BURTI_FILE` and `GABREDARRE_FILE` constants in `arimasktime.py` if your data files are located elsewhere.

## Workflow Overview

### Data Preparation
- Reads and merges river level and weather data.
- Parses dates and handles missing values.
- Calculates the average lag between precipitation peaks and river level peaks.
- Creates lagged features based on the average lag.
- Sets the date index with daily frequency.

### Model Training
- Splits the data into training and testing sets.
- Initializes and fits the AutoARIMA model.
- Saves the trained model to disk.
- Evaluates the model using Mean Squared Error (MSE).

### Forecasting
- Prepares exogenous data for the forecast period.
- Loads the trained model.
- Makes predictions for the specified forecast horizon.

### Visualization
- Plots actual and forecasted river levels for comparison.

## Dependencies

- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `sktime`
- `joblib`
- `logging`

Install them using:

```bash
pip install -r requirements.txt
```

Acknowledgments

- `sktime` for the time series forecasting framework.
- `scikit-learn` for machine learning utilities.
- `matplotlib` for plotting and visualization.