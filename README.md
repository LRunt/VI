# VI
This is an semester work for subject Visualization Information. 
The repository contains interactive Python-based visualization tool for exploring high-frequency Limit Order Book (LOB) and time series data.
The visualization primarily shows the trend and correaltion between time series of financial data.

## Features
- Interactive dashboard built with Dash and Plotly
- Visualization of multidimensional time-series data
- Real-time aggregation of data using configurable time windows
- Multiple aggregation methods support (mean, max, min, sum, etc.)
- Multiple preprocessing techniques:
    - None
    - Normalized
    - Percentage change
- Filtering:
    - By correlation
    - By category
    - Manually
- Visualization of time series:
    - Heatmap
    - 2D Chart (with highlight function)
    - 3D chart (with highlight function)
    - Sparline chart
    - Horizon chart
    - Correlation matrix
    - t-SNE
    - UMAP
- Sorting:
    - By alphabet
    - By correlation

## Instalation

To run the project locally, make sure you have Python 3 installed (Python 3.11 is recommended)

First install the required packages using pip. You can do this by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

## How to run application

To run the application, navigate to the project directory in your terminal and execute:

```bash
python src/viz/visuals.py   # Extension of Dominik Zappe's work. (You must be in folder: )
python visualization.py     # The visualization of data from Yahoo finance. (You must be in folder: sap500_visualization/data_scraping_and_visualization)
```

Or run the batch scripts:

```bash
run_SAP500_visuals.bat
run_zappe_extended_work.bat
```

## Content of project

- doc: Documentation and its source code
- bin: scripts for running the application
- src: source codes of visualizations
    - sap500_visualization/data_scraping_and_visualization - Contains download stock data from Yahoo finance and visualization of these downloaded data
    - preprocessing - Contains script for update data with column mid price
    - timeseries_generation - Constains script for generating random time series data
    - zappe_work_extension - Contains an extension of Dominik Zappe's work with relevant extensions

## Screenshots

