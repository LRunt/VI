import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timedelta


def get_sp500_tickers():
    """Scrapes the current S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 ticker list from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Add a User-Agent to bypass Wikipedia's 403 Forbidden bot-block
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    table = pd.read_html(url, storage_options=headers)[0]

    # Yahoo Finance uses hyphens instead of dots for classes (e.g., BRK.B -> BRK-B)
    tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers


def download_1m_data_for_day(target_date_str):
    """Downloads 1-minute data for all S&P 500 stocks for a specific day."""

    # Validate and parse the date
    try:
        start_date = pd.to_datetime(target_date_str)
        # To get a full day, the end date needs to be the next day
        end_date = start_date + timedelta(days=1)
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    # Check if the date is within the 7-day window allowed by yfinance
    if start_date < (datetime.now() - timedelta(days=7)):
        print("WARNING: yfinance only supports 1-minute data for the last 7 days.")
        print("You may not get any data back for this date.")

    # Create a directory to store the CSVs
    output_dir = f"SP500_1m_data_{target_date_str}"
    os.makedirs(output_dir, exist_ok=True)

    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers. Starting download...")

    for ticker in tickers:
        try:
            # Download the 1-minute data
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1m'
            )

            if df.empty:
                print(f"[{ticker}] No data found for this date.")
                continue

            # Reset index so 'Datetime' becomes a column
            df = df.reset_index()

            # Convert the timezone-aware Datetime to nanosecond Unix timestamp
            df['Time'] = df['Datetime'].astype('int64')

            # We use 'Close' as the Price.
            df['Price'] = df['Close']

            # Filter down to exactly the two columns requested
            final_df = df[['Time', 'Price']]

            # Save to CSV
            filename = os.path.join(output_dir, f"{ticker}.csv")
            final_df.to_csv(filename, index=False)
            print(f"[{ticker}] Saved {len(final_df)} rows to {filename}")

        except Exception as e:
            print(f"[{ticker}] Error downloading data: {e}")


def download_and_save_sectors(output_filename='sp500_sectors.json'):
    """Scrapes Wikipedia for S&P 500 sectors and saves them to a JSON file."""
    try:
        print("Fetching sector categories from Wikipedia...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}

        # Read the first table from the Wikipedia page
        table = pd.read_html(url, storage_options=headers)[0]

        # Yahoo Finance uses hyphens, Wikipedia uses dots (e.g., BRK.B -> BRK-B)
        table['Symbol'] = table['Symbol'].str.replace('.', '-', regex=False)

        # Create a dictionary mapping the Symbol to the GICS Sector
        sector_dict = dict(zip(table['Symbol'], table['GICS Sector']))

        # Save to JSON
        with open(output_filename, 'w') as f:
            json.dump(sector_dict, f, indent=4)

        print(f"Successfully saved {len(sector_dict)} sectors to '{output_filename}'.")

    except Exception as e:
        print(f"Error: Could not fetch or save sectors. ({e})")


if __name__ == "__main__":
    print("--- S&P 500 1-Minute Data Downloader ---")
    user_date = "2026-05-04"#input("Enter the date you want to download (YYYY-MM-DD): ")
    download_1m_data_for_day(user_date)
    download_and_save_sectors("sap500_categories.json")