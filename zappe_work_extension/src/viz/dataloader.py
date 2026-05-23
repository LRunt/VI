import os
import numpy as np
import pandas as pd
import json


# Minute in seconds (60 seconds)
MINUTE_SEC = 60
# Hour in seconds (60 minutes)
HOUR_SEC = 60 * MINUTE_SEC
# Day in seconds (24 hours)
DAY_SEC = 24 * HOUR_SEC


def load_data(date, market_segment_id, security, level_depth=1):
    """
    Load the data from the given date, market segment ID, and security
    :param date: Date of the data to load
    :param market_segment_id: Market segment ID of the data to load
    :param security: Security ID of the data to load
    :param level_depth: Level depth of prices to load (default 1 -- Top of the Book)
    :return: DataFrame with the data
    """
    lobster_fp = f"data/{date}_{market_segment_id}_{security}_lobster_augmented.csv"
    data = pd.read_csv(lobster_fp, sep=",")

    # Throw away Ask Price i and Bid Price i if the ith level > level_depth
    i = level_depth + 1
    while True:
        if f"Ask Price {i}" not in data.columns and f"Bid Price {i}" not in data.columns:
            break
        data = data.drop(columns=[f"Ask Price {i}", f"Bid Price {i}"], errors="ignore")
        i += 1

    return data


def load_ensemble_results(date, market_segment_id, security):
    """
    Load the ensemble results from the given date, market segment ID, and security
    Ensemble meaning results of Isolation Forest, FFNN Autoencoder, CNN Autoencoder, and Transformer Autoencoder
    :param date: Date of the data to load
    :param market_segment_id: Market segment ID of the data to load
    :param security: Security ID of the data to load
    :return: y_pred, y_scores, anomaly_proba
    """
    # Check if the results exist
    res_dir = f"res/{date}_{market_segment_id}_{security}"
    if not os.path.exists(os.path.join(res_dir)):
        return None
    files = os.listdir(res_dir)
    if not files or files == []:
        return None

    # Load all the model predictions
    temp_results = []
    for file in files:
        if file.endswith(".json") and "if_" in file or "ffnn_" in file or "cnn_" in file or "transformer_" in file:
            # Load  the results
            with open(os.path.join(res_dir, file), "r") as fp:
                store = json.load(fp)
                # All we care about is the y_scores -> y_pred and anomaly_proba is calculated based on that
                y_scores = np.array(store["y_scores"])

                # Autoencoder models take anomaly score as loss (lower is better)
                # Scikit-learn models have different anomaly score, where higher is better
                if "if_" in file:
                    y_scores = -y_scores  # Invert the scores, so that we can later use mean across the models

                # Normalize the scores so we can take a "meaningful" average of them
                y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                temp_results.append(y_scores)

    # If there are some results that have different lengths, trim them
    majority_len = np.median([len(x) for x in temp_results])
    temp_results = [x[:int(majority_len)] for x in temp_results]

    # Create the ensemble by averaging the scores
    y_scores_ensemble = np.mean(temp_results, axis=0)


    def transform_ys(y_scores, contamination=0.01, lower_is_better=True):
        """
        Transform the scores to predictions based on expected contamination
        :param y_scores: Y scores
        :param contamination: Contamination
        :param lower_is_better: Lower is better (True for NN, False for Scikit models)
        :return: Y predictions, anomaly probability
        """
        how_many_can_be = len(y_scores) * (1 - contamination)
        y_pred = np.zeros_like(y_scores)

        if lower_is_better:
            y_pred[np.argsort(y_scores)[:int(how_many_can_be)]] = 1
            y_pred[np.argsort(y_scores)[int(how_many_can_be):]] = -1
        else:
            y_pred[np.argsort(y_scores)[::-1][:int(how_many_can_be)]] = 1
            y_pred[np.argsort(y_scores)[::-1][int(how_many_can_be):]] = -1

        y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

        if lower_is_better:
            anomaly_proba = y_scores_norm
        else:
            anomaly_proba = 1 - y_scores_norm

        return y_pred, anomaly_proba


    y_pred_ensemble, anomaly_proba_ensemble = transform_ys(y_scores_ensemble, contamination=0.01, lower_is_better=True)

    # Threshold the ensemble model to only keep the MOST sure predictions of anomalies
    # Keep only the predictions, that are 99.9 % sure or more
    threshold = np.percentile(y_scores_ensemble, 99.9)
    y_pred_ensemble[anomaly_proba_ensemble < threshold] = 1
    anomaly_proba_ensemble[anomaly_proba_ensemble < threshold] = 0

    return y_pred_ensemble, anomaly_proba_ensemble


def load_all_data(level_depth=1):
    """
    Load all data from the data folder
    :param level_depth: Level depth of prices to load (default 1 -- Top of the Book)
    :return: List of DataFrames with the data and a list of names for each DataFrame and results of anomaly detections
    """
    data = []
    names = []
    detections = []

    for file in os.listdir("data"):
        if file.endswith("_lobster_augmented.csv"):
            date, market_segment_id, security = file.split("_")[:3]
            names.append(f"{date}_{market_segment_id}_{security}")
            data.append(load_data(date, market_segment_id, security, level_depth=level_depth))
            detections.append(load_ensemble_results(date, market_segment_id, security))

    return data, names, detections


def aggregate_data(all_data, metric="Ask Price 1", aggregation=np.mean, time_window=3600):
    """
    Aggregate the data in the given time window
    :param all_data: Loaded data in the form of a list of DataFrames
    :param metric: Metric to aggregate
    :param aggregation: Aggregation function to use
    :param time_window: Time window for aggregation (in seconds)
    :return: Aggregated data in the form of a list of numpy arrays
    """
    aggregated_data = []  # Result list

    for data in all_data:
        tmp_agg = []  # Temporary list to store the aggregated data for each time window

        timestamps = pd.to_datetime(data["Time"].values, unit="ns")
        timestamps_series = pd.Series(timestamps)
        seconds_since_midnight = (timestamps_series - timestamps_series.dt.normalize()).dt.total_seconds()

        for i in range(0, DAY_SEC, time_window):
            start_time = i
            end_time = i + time_window

            # Get the data in the time window
            data_in_window = data[(seconds_since_midnight >= start_time) & (seconds_since_midnight < end_time)]
            data_in_window = data_in_window[metric].dropna()

            # Aggregate the metric in the time window
            if data_in_window.empty:  # If there is no data in the time window, append NaN
                tmp_agg.append(np.nan)
                continue
            # Use the aggregation function passed as argument
            tmp_agg.append(aggregation(data_in_window.values))

        # Just convert to numpy array and skip the hardcoded normalization
        tmp_agg = np.array(tmp_agg)
        aggregated_data.append(tmp_agg)

    # Convert to numpy array
    aggregated_data = np.array(aggregated_data)

    return aggregated_data