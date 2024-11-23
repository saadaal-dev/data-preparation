import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.arima import AutoARIMA

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants for file paths and parameters
BULO_BURTI_FILE = "./data/raw/stations/bulo_burti_river_station.csv"
GABREDARRE_FILE = "./data/raw/weather/Ethiopia_Gabredarre_Fafen_river_midpoint_historical_weather_daily_2024-11-19.csv"
DATE_FORMAT_BULO = '%d/%m/%Y'
DATE_FORMAT_GABREDARRE = '%Y-%m-%d'
PEAK_HEIGHT = 1
MODEL_FILENAME_DEFAULT = 'arima_model.pkl'
FORECAST_HORIZON_DEFAULT = 6
DAYS_BEFORE_FORECAST_DEFAULT = 30


def read_csv_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads CSV files for Bulo Burti river station and Gabredarre Fafen river weather data.

    Returns:
        tuple: A tuple containing two pandas DataFrames for Bulo Burti and Gabredarre Fafen data.

    Raises:
        FileNotFoundError: If any of the CSV files are not found.
        pd.errors.ParserError: If there is an error parsing the CSV files.
    """
    try:
        bulo = pd.read_csv(BULO_BURTI_FILE)
        gabredarre = pd.read_csv(GABREDARRE_FILE)
        logging.info("CSV files successfully read.")
        return bulo, gabredarre
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV files: {e}")
        raise


def parse_dates(bulo: pd.DataFrame, gabredarre: pd.DataFrame) -> pd.DataFrame:
    """
    Parses date columns in the provided DataFrames and merges them on the date.

    Args:
        bulo (pd.DataFrame): DataFrame containing Bulo Burti river station data.
        gabredarre (pd.DataFrame): DataFrame containing Gabredarre Fafen weather data.

    Returns:
        pd.DataFrame: Merged DataFrame on the 'date' column with selected features.

    Raises:
        KeyError: If expected columns are missing in the input DataFrames.
    """
    try:
        # Parse 'date' columns to datetime
        bulo['date'] = pd.to_datetime(bulo['date'], format=DATE_FORMAT_BULO)
        gabredarre['date'] = pd.to_datetime(
            gabredarre['date'].str.split().str[0], format=DATE_FORMAT_GABREDARRE
        )
        logging.info("Dates parsed successfully.")

        # Merge DataFrames on 'date'
        merged_data = pd.merge(
            gabredarre[['date', 'precipitation_sum']],
            bulo[['date', 'level(m)']],
            on='date',
            how='inner'
        )
        logging.info("DataFrames merged on 'date'.")
        return merged_data
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        raise


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values by interpolating river levels and forward filling precipitation data.

    Args:
        data (pd.DataFrame): DataFrame containing merged river and precipitation data.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # Interpolate missing river levels linearly
    data['level(m)'] = data['level(m)'].interpolate(method='linear')
    # Forward fill missing precipitation sums
    data['precipitation_sum'] = data['precipitation_sum'].ffill()
    logging.info("Missing values handled.")
    return data


def calculate_average_lag(data: pd.DataFrame) -> int:
    """
    Calculates the average lag between precipitation peaks and river level peaks.

    Args:
        data (pd.DataFrame): DataFrame containing precipitation and river level data.

    Returns:
        int: The average lag in days.

    Raises:
        ValueError: If no peak lags are found.
    """
    # Detect peaks in precipitation and river level
    precipitation_peaks, _ = find_peaks(data['precipitation_sum'], height=PEAK_HEIGHT)
    river_level_peaks, _ = find_peaks(data['level(m)'], height=PEAK_HEIGHT)
    logging.info(f"Found {len(precipitation_peaks)} precipitation peaks and {len(river_level_peaks)} river level peaks.")

    peak_lags = []
    for p_peak in precipitation_peaks:
        # Find the closest river level peak to the current precipitation peak
        closest_river_peak = river_level_peaks[np.abs(river_level_peaks - p_peak).argmin()]
        lag = closest_river_peak - p_peak
        peak_lags.append(lag)
        logging.debug(f"Precipitation peak at {p_peak}, closest river peak at {closest_river_peak}, lag: {lag}")

    if not peak_lags:
        logging.error("No peak lags found.")
        raise ValueError("No peak lags found.")

    average_lag = int(np.mean(peak_lags))
    logging.info(f"Average lag calculated: {average_lag} days.")
    return average_lag


def create_lag_features(data: pd.DataFrame, average_lag: int) -> pd.DataFrame:
    """
    Creates lagged features for precipitation and river levels based on the average lag.

    Args:
        data (pd.DataFrame): DataFrame with precipitation and river level data.
        average_lag (int): The average lag in days to create lagged features.

    Returns:
        pd.DataFrame: DataFrame with new lagged feature columns.
    """
    for lag in range(1, average_lag + 1):
        # Create lagged precipitation features
        data[f'precipitation_sum_day-{lag}'] = data['precipitation_sum'].shift(lag)
        # Create lagged river level features
        data[f'level(m)_day+{lag}'] = data['level(m)'].shift(-lag)
        logging.debug(f"Lagged features created for lag: {lag}")

    # Drop rows with NaN values introduced by lagging
    data = data.dropna().reset_index(drop=True)
    logging.info("Lagged features created and NaNs dropped.")
    return data


def set_date_index(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the 'date' column as the DataFrame index and ensures daily frequency.

    Args:
        data (pd.DataFrame): DataFrame with precipitation and river level data.

    Returns:
        pd.DataFrame: DataFrame indexed by 'date' with daily frequency.

    Raises:
        ValueError: If 'date' column is missing.
    """
    if 'date' not in data.columns:
        logging.error("'date' column is missing from data.")
        raise ValueError("'date' column is missing from data.")

    # Ensure 'date' is datetime and set as index
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    # Set frequency to daily and interpolate missing dates
    data = data.asfreq('D').interpolate(method='time').dropna()
    logging.info("Date index set with daily frequency.")
    return data


def prepare_data() -> tuple[pd.DataFrame, int]:
    """
    Orchestrates the data preparation process by reading, parsing, handling missing values,
    calculating lag, creating lagged features, and setting the date index.

    Returns:
        tuple: A tuple containing the prepared DataFrame and the average lag.

    Raises:
        Exception: Propagates any exception that occurs during the data preparation steps.
    """
    try:
        # Read CSV files
        bulo, gabredarre = read_csv_files()

        # Parse dates and merge DataFrames
        data = parse_dates(bulo, gabredarre)

        # Handle missing values
        data = handle_missing_values(data)

        # Calculate average lag between precipitation and river level peaks
        average_lag = calculate_average_lag(data)

        # Create lagged feature columns based on average lag
        data = create_lag_features(data, average_lag)

        # Set 'date' as index with daily frequency
        data = set_date_index(data)

        logging.info("Data preparation completed successfully.")
        return data, average_lag
    except Exception as e:
        logging.error(f"Error in preparing data: {e}")
        raise


def train_model(data: pd.DataFrame, average_lag: int, model_filename: str = 'arima_model', forecast_horizon: int = 6) -> AutoARIMA:
    """
    Trains the AutoARIMA model using the prepared data and saves the trained model to disk.

    Parameters
    ----------
    data : pd.DataFrame
        The prepared data with lagged features.
    average_lag : int
        The average lag between precipitation and river level peaks.
    model_filename : str, optional
        The filename to save the trained model. Default is 'arima_model'.
    forecast_horizon : int, optional
        The number of days to forecast. Default is 6.

    Returns
    -------
    forecaster : AutoARIMA
        The trained AutoARIMA forecaster.

    Raises
    ------
    Exception
        If model training or saving fails.
    """
    try:
        # Prepare the endogenous (target) variable and exogenous variables
        exog_vars = ['precipitation_sum'] + [
            f'precipitation_sum_day-{lag}' for lag in range(1, average_lag + 1)
        ]

        # Extract the endogenous and exogenous data
        endog = data['level(m)']
        exog = data[exog_vars]

        # Split the data into training and testing sets
        train_endog = endog.iloc[:-forecast_horizon]
        train_exog = exog.iloc[:-forecast_horizon]

        test_endog = endog.iloc[-forecast_horizon:]
        test_exog = exog.iloc[-forecast_horizon:]

        # Prepare the forecasting horizon
        fh = ForecastingHorizon(test_endog.index, is_relative=False)

        # Initialize and fit the forecaster
        forecaster = AutoARIMA(
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None,
            seasonal=False,
            suppress_warnings=True,
            stepwise=True,
            error_action='ignore',
            trace=True
        )
        forecaster.fit(y=train_endog, X=train_exog)

        model_filename = f'{model_filename}.pkl'

        # Save the trained model to disk
        joblib.dump(forecaster, model_filename)
        logging.info(f"Model saved to {model_filename}")

        # Evaluate the model
        forecast = forecaster.predict(fh=fh, X=test_exog)
        mse = mean_squared_error(test_endog, forecast)
        logging.info(f'Mean Squared Error on Test Set: {mse}')

        return forecaster
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise


def load_model_and_forecast(new_exog: pd.DataFrame, model_filename: str = 'arima_model') -> pd.Series:
    """
    Loads the trained model from disk and makes forecasts using new exogenous data.

    Parameters
    ----------
    new_exog : pd.DataFrame
        The new exogenous data for making forecasts.
    model_filename : str, optional
        The filename to load the trained model from. Default is 'arima_model'.

    Returns
    -------
    forecast : pd.Series
        The forecasted river level values.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    Exception
        If forecasting fails.
    """
    try:
        # Load the trained model
        forecaster = joblib.load(f'{model_filename}.pkl')
        logging.info(f"Model loaded from {model_filename}.pkl")

        # Prepare the forecasting horizon
        fh = ForecastingHorizon(new_exog.index, is_relative=False)

        # Make predictions
        forecast = forecaster.predict(fh=fh, X=new_exog)

        return forecast
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error in forecasting: {e}")
        raise

def prepare_exogenous_data_for_forecast(data: pd.DataFrame, forecast_start_date: str, forecast_end_date: str, average_lag: int) -> pd.DataFrame:
    """
    Prepares the exogenous data for the forecast period.

    Parameters
    ----------
    data : pd.DataFrame
        The original data with 'precipitation_sum'.
    forecast_start_date : str
        The start date of the forecast period.
    forecast_end_date : str
        The end date of the forecast period.
    average_lag : int
        The average lag to create lagged features.

    Returns
    -------
    new_exog : pd.DataFrame
        Exogenous data with lagged features for the forecast period.
    """
    # Define the forecast dates
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')

    # Calculate the required start date to include necessary lagged data
    lagged_start_date = pd.to_datetime(forecast_start_date) - pd.Timedelta(days=average_lag)
    required_dates = pd.date_range(start=lagged_start_date, end=forecast_end_date, freq='D')

    # Ensure 'data' contains the necessary precipitation data
    exog_data_full = data['precipitation_sum'].reindex(required_dates)

    # Handle missing values by forward-filling
    exog_data_full.fillna(method='ffill', inplace=True)

    # Create new exogenous DataFrame
    new_exog = pd.DataFrame(index=forecast_dates)
    new_exog['precipitation_sum'] = exog_data_full.loc[forecast_dates]

    # Create lagged features
    for lag in range(1, average_lag + 1):
        new_exog[f'precipitation_sum_day-{lag}'] = exog_data_full.shift(lag).loc[forecast_dates]

    # Drop any rows with missing values
    new_exog.dropna(inplace=True)

    return new_exog

def forecast_river_levels(model_filename: str, forecast_start_date: str, forecast_end_date: str, data: pd.DataFrame = None, average_lag: int = None) -> pd.Series:
    """
    Forecasts river levels for a specified period using an existing model.

    Parameters
    ----------
    model_filename : str
        The filename of the trained model without the '.pkl' extension.
    forecast_start_date : str
        The start date of the forecast period (format 'YYYY-MM-DD').
    forecast_end_date : str
        The end date of the forecast period (format 'YYYY-MM-DD').
    data : pd.DataFrame, optional
        The original data with 'precipitation_sum'. Required to prepare exogenous data.
    average_lag : int, optional
        The average lag to create lagged features.

    Returns
    -------
    forecast : pd.Series
        Forecasted river levels indexed by date.

    Raises
    ------
    Exception
        If forecasting fails.
    """
    try:
        if data is None or average_lag is None:
            # Prepare data to get 'data' and 'average_lag'
            data, average_lag = prepare_data()  # Ensure this function is defined elsewhere

        # Prepare the exogenous data
        new_exog = prepare_exogenous_data_for_forecast(
            data, forecast_start_date, forecast_end_date, average_lag
        )

        # Load the trained model
        forecaster = joblib.load(f'{model_filename}.pkl')
        logging.info(f"Model loaded from {model_filename}.pkl")

        # Prepare the forecasting horizon
        fh = ForecastingHorizon(new_exog.index, is_relative=False)

        # Make predictions
        forecast = forecaster.predict(fh=fh, X=new_exog)

        return forecast
    except Exception as e:
        logging.error(f"An error occurred during forecasting: {e}")
        raise

def plot_forecast(forecast: pd.Series, actual_river_levels: pd.Series = None, days_before_forecast: int = 30):
    """
    Plots the actual and forecasted river levels.

    Parameters
    ----------
    forecast : pd.Series
        The forecasted river level data.
    actual_river_levels : pd.Series, optional
        The actual river level data. Default is None.
    days_before_forecast : int, optional
        The number of days before the forecast to include in the plot. Default is 30.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 6))

    # Determine the start date for plotting
    if actual_river_levels is not None and not actual_river_levels.empty:
        start_plot_date = actual_river_levels.index[0]
        
        # Plot actual river levels
        plt.plot(
            actual_river_levels.index, actual_river_levels,
            label='Actual River Levels', color='blue'
        )
    else:
        # If no actual data, start plot from forecast start date
        start_plot_date = forecast.index[0]

    # Plot forecasted river levels
    plt.plot(
        forecast.index, forecast,
        label='Forecasted River Levels', color='red', linestyle='--'
    )

    # Formatting the dates on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Adding titles and labels
    plt.title('River Level Forecast')
    plt.xlabel('Date')
    plt.ylabel('River Level (m)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the data preparation, model training, forecasting, and plotting.
    """
    try:
        # Prepare data
        data, average_lag = prepare_data()

        # Train the model and save it
        forecaster = train_model(
            data,
            average_lag,
            model_filename=MODEL_FILENAME_DEFAULT.replace('.pkl', ''),
            forecast_horizon=FORECAST_HORIZON_DEFAULT
        )

        # Define forecast dates based on the last 'forecast_horizon' days
        forecast_start_date = data.index[-FORECAST_HORIZON_DEFAULT].strftime('%Y-%m-%d')
        forecast_end_date = data.index[-1].strftime('%Y-%m-%d')

        # Forecast river levels using the refactored function
        forecast = forecast_river_levels(
            model_filename=MODEL_FILENAME_DEFAULT.replace('.pkl', ''),
            forecast_start_date=forecast_start_date,
            forecast_end_date=forecast_end_date,
            data=data,
            average_lag=average_lag
        )

        # Optionally, plot the forecast using the updated plot_forecast function
        actual_river_levels = data['level(m)']
        plot_forecast(
            forecast=forecast,
            actual_river_levels=actual_river_levels,
            days_before_forecast=DAYS_BEFORE_FORECAST_DEFAULT
        )

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()