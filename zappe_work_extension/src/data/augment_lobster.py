import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import time
import numpy as np
import pandas as pd
import datetime

# Default values
MARKET_ID = "XEUR"
DATE = "20191202"
MARKET_SEGMENT_ID = "688"
SECURITY_ID = "4128839"

# User defined values
if len(sys.argv) == 5:
    MARKET_ID = sys.argv[1]
    DATE = sys.argv[2]
    MARKET_SEGMENT_ID = sys.argv[3]
    SECURITY_ID = sys.argv[4]

# Input and output file paths
INPUT_FILE_PATH = f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_lobster.csv"
OUTPUT_FILE_PATH = f"data/{DATE}_{MARKET_SEGMENT_ID}_{SECURITY_ID}_lobster_augmented.csv"

# Add imbalance index, frequency of incoming messages, cancellations rate, etc. to the CSV
print("Augmenting CSV with extra features")
print("Extracting features...")
tic = time.time()


def get_imbalance_index(data, alpha=0.5, levels=3):
    """
    Calculate imbalance index for a given orderbook.
    :param data : Dataframe with the orderbook data
    :param alpha: parameter for imbalance index
    :param levels: number of levels to consider
    :return: imbalance index
    """
    assert alpha > 0, "Alpha must be positive"
    assert levels > 0, "Level must be positive"

    ask_columns = [f'Ask Volume {i}' for i in range(1, levels + 1)]
    bid_columns = [f'Bid Volume {i}' for i in range(1, levels + 1)]
    lobster_data_matrix = data[ask_columns + bid_columns].values
    asks = lobster_data_matrix[:, :levels]
    bids = lobster_data_matrix[:, levels:]

    assert asks.shape[1] >= levels and bids.shape[1] >= levels, "Not enough levels in orderbook"

    # Calculate imbalance index
    V_bt = np.sum(bids[:, :levels] * np.exp(-alpha * np.arange(0, levels)), axis=1)
    V_at = np.sum(asks[:, :levels] * np.exp(-alpha * np.arange(0, levels)), axis=1)
    return (V_bt - V_at) / (V_bt + V_at)


def get_frequency_of_all_incoming_actions(timestamps, time=300):
    """
    Returns the frequency of incoming actions (messages) in the last *time* seconds for all timestamps
    :param timestamps: Array of timestamps (in nano seconds)
    :param time: Time window in seconds (default value is 300)
    :return: List of frequencies of incoming actions (messages) for all timestamps
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Initialize an array to hold the frequencies
    freqs = np.zeros_like(timestamps, dtype=float)

    # Calculate the frequency for each timestamp
    for start in range(len(timestamps) - 1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        freqs[start] = start - end

    return freqs / time


def get_cancelations_rate(data, timestamps, time=300):
    """
    Returns the cancellations rate in the last *time* seconds for all timestamps
    :param data: Dataframe with the orderbook data
    :param timestamps: Array of timestamps
    :param time: Time window in seconds (default value is 300)
    :return: List of cancellations rate for all timestamps
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Prepare the data
    cancellations = data["Cancellations Buy"].values + data["Cancellations Sell"].values

    # Initialize an array to hold the cancellations rate
    cancellation_rate = np.zeros_like(timestamps, dtype=float)

    # Calculate the cancellations rate for each timestamp
    for start in range(len(timestamps) - 1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        cancellation_rate[start] = np.sum(cancellations[end:start])

    return cancellation_rate / time


def get_high_quoting_activity(data, timestamps, levels=5, time=1):
    """
    High Quoting Activity
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Prepare the data
    ask_volume_names = [f"Ask Volume {i}" for i in range(1, levels + 1)]
    bid_volume_names = [f"Bid Volume {i}" for i in range(1, levels + 1)]
    ask_volumes = np.array([data[name].values for name in ask_volume_names])
    bid_volumes = np.array([data[name].values for name in bid_volume_names])
    entry_ask_volumes = ask_volumes[:, 1:] - ask_volumes[:, :-1]
    entry_bid_volumes = bid_volumes[:, 1:] - bid_volumes[:, :-1]

    # Add zeros to the beginning of the arrays
    entry_ask_volumes = np.insert(entry_ask_volumes, 0, 0, axis=1)
    entry_bid_volumes = np.insert(entry_bid_volumes, 0, 0, axis=1)

    # Initialize an array to hold the high quoting activities
    high_quoting_activity = np.zeros_like(timestamps, dtype=float)

    # Calculate the high quoting activity for each timestamp
    for start in range(len(timestamps) - 1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        if end < start and start - end > 0:
            ask_bid_sum = ask_volumes[:, end:start] + bid_volumes[:, end:start]
            if ask_bid_sum.size > 0:
                ask_bid_sum[ask_bid_sum == 0] = np.nan
                high_quoting_activity[start] = np.nanmax(
                    np.abs(entry_ask_volumes[:, end:start] - entry_bid_volumes[:, end:start]) / ask_bid_sum)
            else:
                high_quoting_activity[start] = 0
        else:
            high_quoting_activity[start] = 0

    return high_quoting_activity


def get_unbalanced_quoting(data, timestamps, levels=5, time=1):
    """
    Unbalanced Quoting
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Prepare the data
    ask_volume_names = [f"Ask Volume {i}" for i in range(1, levels + 1)]
    bid_volume_names = [f"Bid Volume {i}" for i in range(1, levels + 1)]
    ask_volumes = np.array([data[name].values for name in ask_volume_names])
    bid_volumes = np.array([data[name].values for name in bid_volume_names])

    # Initialize an array to hold the unbalanced quoting activities
    unbalanced_quoting = np.zeros_like(timestamps, dtype=float)

    # Calculate the unbalanced quoting activity for each timestamp
    for start in range(len(timestamps) - 1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        if end < start and start - end > 0:
            ask_bid_sum = ask_volumes[:, end:start] + bid_volumes[:, end:start]
            if ask_bid_sum.size > 0:
                ask_bid_sum[ask_bid_sum == 0] = np.nan
                unbalanced_quoting[start] = np.nanmax(
                    np.abs(ask_volumes[:, end:start] - bid_volumes[:, end:start]) / ask_bid_sum)
            else:
                unbalanced_quoting[start] = 0
        else:
            unbalanced_quoting[start] = 0

    return unbalanced_quoting


def get_low_execution_proba(data, timestamps, levels=5, time=1):
    """
    Low Execution Probability
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Prepare the data
    ask_volume_names = [f"Ask Volume {i}" for i in range(1, levels + 1)]
    bid_volume_names = [f"Bid Volume {i}" for i in range(1, levels + 1)]
    ask_volumes = np.array([data[name].values for name in ask_volume_names])
    bid_volumes = np.array([data[name].values for name in bid_volume_names])
    ask_volumes_2_5 = np.sum(ask_volumes[1:], axis=0)
    bid_volumes_2_5 = np.sum(bid_volumes[1:], axis=0)

    # Initialize an array to hold the low execution probabilities
    low_execution_proba = np.zeros_like(timestamps, dtype=float)

    # Calculate the low execution probabilities for each timestamp
    for start in range(len(timestamps) - 1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        if end < start and start - end > 0:
            ask_bid_sum = ask_volumes[:, end:start] + bid_volumes[:, end:start]
            if ask_bid_sum.size > 0:
                ask_bid_sum[ask_bid_sum == 0] = np.nan
                low_execution_proba[start] = np.nanmax(
                    np.abs(ask_volumes_2_5[end:start] - bid_volumes_2_5[end:start]) / ask_bid_sum)
            else:
                low_execution_proba[start] = 0
        else:
            low_execution_proba[start] = 0

    return low_execution_proba


def get_trades_oppose_quotes(data, timestamps, levels=5, time=1):
    """
    Trades Oppose Quotes
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Prepare the data
    trade_on_buy = data["Trades Buy"].values
    trade_on_sell = data["Trades Sell"].values

    # Get the imbalance indices
    imbalance_indcs = get_imbalance_index(data, alpha=0.5, levels=levels)

    # Initialize an array to hold the trades oppose quotes
    trades_oppose_quotes = np.zeros_like(timestamps, dtype=int)

    # Calculate the trades oppose quotes for each timestamp
    for start in range(len(timestamps) - 1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        if end < start and start - end > 0:
            if imbalance_indcs[end] > 0.1 and trade_on_sell[start] != 0:
                trades_oppose_quotes[start] = 1
            elif imbalance_indcs[end] < -0.1 and trade_on_buy[start] != 0:
                trades_oppose_quotes[start] = 1
            else:
                trades_oppose_quotes[start] = 0

    return trades_oppose_quotes


def get_cancels_oppose_trades(data, timestamps, levels=5, time=1):
    """
    Cancels Oppose Trades
    """
    # Convert time to nanoseconds
    time_ns = time * 1e9

    # Initialize two pointers at the end of the timestamps array
    start, end = len(timestamps) - 1, len(timestamps) - 1

    # Prepare the data
    trade_on_buy = data["Trades Buy"].values
    trade_on_sell = data["Trades Sell"].values

    # Get the cancellation rates
    cancellation_rate = get_cancelations_rate(data, timestamps, time=time)
    thresh = np.percentile(cancellation_rate, 10)

    # Initialize an array to hold the cancels oppose trades
    cancels_oppose_trades = np.zeros_like(timestamps, dtype=int)

    # Calculate the cancels oppose trades for each timestamp
    for start in range(len(timestamps) - 1, -1, -1):
        while end >= 0 and timestamps[start] - timestamps[end] <= time_ns:
            end -= 1
        if end < start and start - end > 0:
            if cancellation_rate[end] > thresh and trade_on_sell[start] != 0:
                cancels_oppose_trades[start] = 1
            elif cancellation_rate[end] > thresh and trade_on_buy[start] != 0:
                cancels_oppose_trades[start] = 1
            else:
                cancels_oppose_trades[start] = 0

    return cancels_oppose_trades


def get_mid_price(data):
    """
    Calculate the Mid Price from the 1st level of the orderbook.
    :param data: Dataframe with the orderbook data
    :return: List/Array of mid prices
    """
    # Extracts the best Ask and best Bid
    ask_price_1 = data["Ask Price 1"].values
    bid_price_1 = data["Bid Price 1"].values

    # Calculates the average
    mid_prices = (ask_price_1 + bid_price_1) / 2.0

    return mid_prices


# ==========================================
# MAIN EXECUTION
# ==========================================

data = pd.read_csv(INPUT_FILE_PATH, delimiter=",")
timestamps = np.array(data["Time"])

# Extract Mid Price
mid_prices = get_mid_price(data)

# Imbalance index
imbalance_indices = get_imbalance_index(data, alpha=0.5, levels=30)

# Frequency of incoming messages
freqs = get_frequency_of_all_incoming_actions(timestamps, time=300)

# Cancellations rate
cancellation_rate = get_cancelations_rate(data, timestamps, time=300)

# High Quoting Activity
high_quoting_activity = get_high_quoting_activity(data, timestamps, levels=5, time=1)

# Unbalanced Quoting
unbalanced_quoting = get_unbalanced_quoting(data, timestamps, levels=5, time=1)

# Low Execution Probability
low_execution_proba = get_low_execution_proba(data, timestamps, levels=5, time=1)

# Trades Oppose Quotes
trades_oppose_quotes = get_trades_oppose_quotes(data, timestamps, levels=5, time=1)

# Cancels Oppose Trades
cancels_oppose_trades = get_cancels_oppose_trades(data, timestamps, levels=5, time=1)

# Add the new columns to the CSV and drop the old "raw" ones
data = data.drop(columns=["Cancellations Buy", "Cancellations Sell"])  # Drop raw cancellations column
data = data.drop(columns=["Trades Buy", "Trades Sell"])  # Drop raw trades columns

data["Mid Price"] = mid_prices
data["Imbalance Index"] = imbalance_indices
data["Frequency of Incoming Messages"] = freqs
data["Cancellations Rate"] = cancellation_rate
data["High Quoting Activity"] = high_quoting_activity
data["Unbalanced Quoting"] = unbalanced_quoting
data["Low Execution Probability"] = low_execution_proba
data["Trades Oppose Quotes"] = trades_oppose_quotes
data["Cancels Oppose Trades"] = cancels_oppose_trades

# Filter out the timestamps that are not from the correct date
start_nanosec = datetime.datetime.strptime(DATE, "%Y%m%d").timestamp() * 1e9
end_nanosec = start_nanosec + 24 * 60 * 60 * 1e9
data = data[(timestamps >= start_nanosec) & (timestamps <= end_nanosec)]

tac = time.time()
print(f"Features extracted in {tac - tic:.2f} seconds")

# Export to CSV
tic = time.time()
print("Exporting augmented CSV...")

data.to_csv(OUTPUT_FILE_PATH, index=False)

tac = time.time()
print(f"Augmented CSV exported in {tac - tic:.2f} seconds")