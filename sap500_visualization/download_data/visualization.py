import os
import json
import math
import numpy as np
import pandas as pd
import datetime

import plotly.express as px
import plotly.graph_objects as go
import plotly_resampler
from plotly_resampler import FigureResampler

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# --- Optional import for UMAP ---
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: 'umap-learn' is not installed. UMAP clustering will be disabled. Run 'pip install umap-learn' to enable.")

# --- Optional import for t-SNE ---
try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: 'scikit-learn' is not installed. t-SNE clustering will be disabled. Run 'pip install scikit-learn' to enable.")

# --- Data Loading Logic ---

MINUTE_SEC = 60
HOUR_SEC = 60 * MINUTE_SEC

# Global variables for categorization
ticker_to_sector = {}
sorted_sectors = []


def find_data_directory():
    """Automatically find the directory created by the previous scrape script."""
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("SP500_1m_data_"):
            return item
    return "data"


def load_all_data():
    """Loads all S&P 500 1-minute CSV files."""
    data = []
    names = []
    data_dir = find_data_directory()

    if not os.path.exists(data_dir):
        print(f"Error: Could not find the folder '{data_dir}'. Make sure you run the scraping script first.")
        return [], []

    print(f"Loading data from folder: {data_dir}...")
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            ticker = file.replace(".csv", "")
            names.append(ticker)
            filepath = os.path.join(data_dir, file)
            df = pd.read_csv(filepath)
            data.append(df)

    return data, names


def fetch_sectors(filename='sap500_categories.json'):
    """Loads the ticker-to-sector mapping from a local JSON file."""
    try:
        if os.path.exists(filename):
            print(f"Loading sector categories from '{filename}'...")
            with open(filename, 'r') as f:
                sector_dict = json.load(f)
            return sector_dict
        else:
            print(f"Warning: '{filename}' not found. Defaulting to 'Unknown' for all sectors.")
            print("Please run the download_sectors.py script first.")
            return {}
    except Exception as e:
        print(f"Warning: Could not read '{filename}'. Defaulting to 'Unknown'. ({e})")
        return {}


def aggregate_data(all_data, metric="Price", aggregation=np.mean, time_window_sec=3600):
    """Aggregates the stock data using Pandas Resample."""
    aggregated_data = []
    minutes = max(1, int(time_window_sec // 60))
    freq = f"{minutes}min"

    # Ensure standard US Market Zone conversion for all aggregated times
    all_times = pd.concat([pd.to_datetime(df['Time'], unit='s') for df in all_data])
    all_times = all_times.dt.tz_localize('UTC').dt.tz_convert('America/New_York')

    if all_times.empty:
        return np.array([]), []

    global_min = all_times.min().floor(freq)
    # Use floor instead of ceil to avoid creating a future empty bucket that gets flatlined by ffill
    global_max = all_times.max().floor(freq)
    master_index = pd.date_range(start=global_min, end=global_max, freq=freq)

    for df in all_data:
        df_temp = df.copy()
        df_temp['Datetime'] = pd.to_datetime(df_temp['Time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
            'America/New_York')
        df_temp.set_index('Datetime', inplace=True)

        resampled = df_temp[metric].resample(freq).apply(aggregation)
        resampled = resampled.reindex(master_index)
        resampled = resampled.ffill().bfill()
        aggregated_data.append(resampled.values)

    master_x_labels = [dt.strftime("%H:%M") for dt in master_index]
    return np.array(aggregated_data), master_x_labels


# --- Dashboard Logic ---

HOST_ADDRESS = "127.0.0.1"
PORT = 8080

timestamps_graph_labels = None
last_update_bottom_graph_click_count = None
last_graph_type = "Heatmap"
last_data_processing_state = "none"
last_sort_by_state = "Default"

chosen_aggregation = "Mean"
aggregation_functions_map = {"Mean": np.mean, "Median": np.median, "Max": np.max, "Min": np.min, "Std": np.std}
metric = "Price"
metric_descriptions_map = {"Price": "The Close price of the stock."}
time_window_aggregation_minutes = 60


def normalize_data(z_data):
    normalized_z = []
    for row in z_data:
        arr = np.array(row, dtype=np.float64)
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if max_val > min_val:
            norm_arr = (arr - min_val) / (max_val - min_val)
        else:
            norm_arr = np.zeros_like(arr) + 0.5
        normalized_z.append(norm_arr)
    return normalized_z


def pct_change_data(z_data):
    pct_z = []
    for row in z_data:
        arr = np.array(row, dtype=np.float64)
        valid_idx = np.where(~np.isnan(arr))[0]
        if len(valid_idx) > 0:
            base_val = arr[valid_idx[0]]
            if base_val == 0:
                pct_arr = np.zeros_like(arr)
            else:
                pct_arr = ((arr - base_val) / np.abs(base_val)) * 100
        else:
            pct_arr = np.zeros_like(arr)
        pct_z.append(pct_arr)
    return pct_z


def calculate_nice_ticks(vmin, vmax, target_ticks=5):
    """Calculates padded min, max, and clean, significant interval ticks (e.g., 0, 2, 4, 6)."""
    if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
        return vmin, vmax, [vmin], [f"{vmin:.2f}"]

    span = vmax - vmin
    rough_step = span / (target_ticks - 1)
    mag = 10 ** math.floor(math.log10(rough_step)) if rough_step > 0 else 1
    rel_step = rough_step / mag

    # Define "significant" jumps based on magnitude
    if rel_step < 1.5:
        nice_step = 1 * mag
    elif rel_step < 3.5:
        nice_step = 2 * mag
    elif rel_step < 7.5:
        nice_step = 5 * mag
    else:
        nice_step = 10 * mag

    # Pad out the min and max limits so the legend includes the extremes cleanly
    padded_min = math.floor(vmin / nice_step) * nice_step
    padded_max = math.ceil(vmax / nice_step) * nice_step

    ticks = np.arange(padded_min, padded_max + nice_step * 0.1, nice_step).tolist()

    def format_tick(val):
        val = round(val, 6)  # Prevent visual floating point artifacts
        if val.is_integer():
            return str(int(val))
        elif nice_step >= 0.1:
            return f"{val:.1f}"
        elif nice_step >= 0.01:
            return f"{val:.2f}"
        else:
            return f"{val:.4f}"

    labels = [format_tick(t) for t in ticks]
    return padded_min, padded_max, ticks, labels


def create_price_graph(timestamps, prices, name, how_many_x_ticks=15):
    global timestamps_graph_labels

    dt_index = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('America/New_York')
    timestamps_graph_labels = dt_index.strftime("%H:%M:%S").tolist()

    timestamps_graph = list(range(len(timestamps_graph_labels)))
    tickvals = list(range(0, len(timestamps), max(1, len(timestamps) // how_many_x_ticks)))
    ticklabels = [timestamps_graph_labels[i] for i in tickvals]

    price_graph_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))
    price_graph_fig.add_trace(
        go.Scattergl(name="Price", yaxis="y1"),
        hf_x=timestamps_graph, hf_y=prices,
        hf_marker_color="rgb(31, 119, 180)"
    )

    price_graph_fig.update_layout(
        title=f"{name} - Intraday Price History",
        xaxis={"title": "Timestamp", "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels,
               "range": [0, len(timestamps_graph_labels) - 1]},
        yaxis={"title": "Price (USD)", "side": "left"},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "center", "x": 0.5, "itemclick": False,
                "itemdoubleclick": False},
        clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9", margin=dict(b=60)
    )

    for trace in price_graph_fig.data:
        trace.name = trace.name.split("~")[0].strip().replace("[R]", "").strip()

    return price_graph_fig


def create_bottom_figure(z_data, x_data, y_names, graph_type, metric_name, agg_name, data_processing="none",
                         tsne_perp=30, umap_neigh=15, umap_dist=0.1):
    fig = go.Figure()
    title_text = f"{agg_name} of {metric_name} ({graph_type})"

    if data_processing == "normalize":
        title_text += " - Normalized Values"
        value_title = "Normalized Value"
    elif data_processing == "pct_change":
        title_text += " - Percentage Change (%)"
        value_title = "Percentage Change (%)"
    else:
        value_title = "Price (USD)"

    num_series = max(1, len(y_names))
    if graph_type == "Heatmap":
        fig_height = max(400, num_series * 20 + 150)
    elif graph_type in ["Horizon Chart", "Spark Line"]:
        fig_height = max(500, num_series * 35 + 150)
    else:
        fig_height = 700

    if graph_type == "Heatmap":
        # Calculate dynamic ticks to force significant min/max display
        valid_z = np.array(z_data, dtype=float)
        valid_z = valid_z[~np.isnan(valid_z)]
        z_min = np.nanmin(valid_z) if len(valid_z) else 0
        z_max = np.nanmax(valid_z) if len(valid_z) else 1

        padded_min, padded_max, tickvals, ticktext = calculate_nice_ticks(z_min, z_max, target_ticks=5)

        fig.add_trace(go.Heatmap(
            z=z_data,
            x=x_data,
            y=y_names,
            colorscale="Viridis",
            zmin=padded_min,
            zmax=padded_max,
            colorbar=dict(
                title=value_title,
                thickness=15,
                outlinewidth=0,
                lenmode="pixels",
                len=250,
                yanchor="top",
                y=1,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext
            ),
            hoverongaps=False
        ))
        fig.update_layout(title=title_text, height=fig_height,
                          xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
                          yaxis={"title": "Stock Ticker", "range": [-0.5, max(1, len(y_names)) - 0.5]},
                          clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9")

    elif graph_type == "Correlation Matrix":
        z_safe = np.nan_to_num(z_data, nan=0.0) + np.random.normal(0, 1e-9, np.array(z_data).shape)
        corr_matrix = np.corrcoef(z_safe)

        fig.add_trace(go.Heatmap(
            z=corr_matrix,
            x=y_names,
            y=y_names,
            colorscale="RdBu",
            zmin=-1, zmax=1, zmid=0,
            colorbar=dict(
                title="Correlation",
                tickmode="array",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
            ),
            hoverongaps=False,
            hovertemplate="Stock X: %{x}<br>Stock Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Correlation Matrix of {metric_name} ({agg_name})",
            height=800,
            xaxis={"title": "Stock Ticker", "tickangle": -45},
            yaxis={"title": "Stock Ticker", "autorange": "reversed"},
            plot_bgcolor="#f9f9f9",
            margin=dict(l=80, b=80)
        )

    elif graph_type == "UMAP Clusters":
        if not UMAP_AVAILABLE:
            fig.add_annotation(text="Missing Library: pip install umap-learn", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="red"))
            return fig

        if len(y_names) < 3:
            fig.add_annotation(text="Please select at least 3 stocks for UMAP clustering.", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=18))
            return fig

        z_safe = np.nan_to_num(z_data, nan=0.0)
        n_neighbors = min(umap_neigh, max(2, len(y_names) - 1))

        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=umap_dist, random_state=42)
        embedding = reducer.fit_transform(z_safe)

        unique_sectors = sorted(list(set([ticker_to_sector.get(name, 'Unknown') for name in y_names])))

        for sector in unique_sectors:
            sector_indices = [idx for idx, name in enumerate(y_names) if
                              ticker_to_sector.get(name, 'Unknown') == sector]

            fig.add_trace(go.Scatter(
                x=embedding[sector_indices, 0],
                y=embedding[sector_indices, 1],
                mode="markers+text",
                name=sector,
                text=[y_names[idx] for idx in sector_indices],
                textposition="top center",
                textfont=dict(size=10, color="rgba(0,0,0,0.6)"),
                marker=dict(size=12, line=dict(width=1, color='White')),
                customdata=[y_names[idx] for idx in sector_indices],
                hovertemplate="<b>%{customdata}</b><br>Sector: " + sector + "<br>UMAP-1: %{x:.2f}<br>UMAP-2: %{y:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title=f"UMAP Projection (neighbors={n_neighbors}, min_dist={umap_dist:.2f})",
            height=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            plot_bgcolor="#f9f9f9",
            hovermode="closest",
            legend=dict(title="Sectors", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, itemclick=False,
                        itemdoubleclick=False)
        )

    elif graph_type == "t-SNE Clusters":
        if not SKLEARN_AVAILABLE:
            fig.add_annotation(text="Missing Library: pip install scikit-learn", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="red"))
            return fig

        if len(y_names) < 3:
            fig.add_annotation(text="Please select at least 3 stocks for t-SNE clustering.", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=18))
            return fig

        z_safe = np.nan_to_num(z_data, nan=0.0)

        perplexity_val = min(tsne_perp, max(1, len(y_names) - 1))

        tsne_model = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca', learning_rate='auto')
        embedding = tsne_model.fit_transform(z_safe)

        unique_sectors = sorted(list(set([ticker_to_sector.get(name, 'Unknown') for name in y_names])))

        for sector in unique_sectors:
            sector_indices = [idx for idx, name in enumerate(y_names) if
                              ticker_to_sector.get(name, 'Unknown') == sector]

            fig.add_trace(go.Scatter(
                x=embedding[sector_indices, 0],
                y=embedding[sector_indices, 1],
                mode="markers+text",
                name=sector,
                text=[y_names[idx] for idx in sector_indices],
                textposition="top center",
                textfont=dict(size=10, color="rgba(0,0,0,0.6)"),
                marker=dict(size=12, line=dict(width=1, color='White')),
                customdata=[y_names[idx] for idx in sector_indices],
                hovertemplate="<b>%{customdata}</b><br>Sector: " + sector + "<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title=f"t-SNE Projection (perplexity={perplexity_val})",
            height=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            plot_bgcolor="#f9f9f9",
            hovermode="closest",
            legend=dict(title="Sectors", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, itemclick=False,
                        itemdoubleclick=False)
        )

    elif graph_type == "3D Lines":
        # vibrant_palette combines several distinct palettes while explicitly removing
        # the trailing dark grey/black/brown colors from Vivid, Prism, and Set1
        vibrant_palette = (
                                  px.colors.qualitative.Plotly +
                                  px.colors.qualitative.Vivid[:-1] +
                                  px.colors.qualitative.Prism[:-1] +
                                  px.colors.qualitative.Set1[:-2] +
                                  px.colors.qualitative.Set2 +
                                  px.colors.qualitative.Set3
                          ) * 10

        for i, name in enumerate(y_names):
            fig.add_trace(
                go.Scatter3d(
                    x=x_data,
                    y=[name] * len(x_data),
                    z=z_data[i],
                    name=name,
                    mode="lines",
                    line=dict(width=4, color=vibrant_palette[i]),
                    customdata=[name] * len(x_data)
                )
            )
        fig.update_layout(title=title_text, height=fig_height,
                          scene=dict(xaxis=dict(title="Time", range=[len(x_data) - 0.5, -0.5], autorange="reversed"),
                                     yaxis=dict(title="Stock Ticker"), zaxis=dict(title=value_title)),
                          clickmode="event+select", plot_bgcolor="#f9f9f9", margin=dict(l=0, r=0, b=0, t=40),
                          showlegend=True,
                          legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02,
                                  "itemclick": False, "itemdoubleclick": False})

    elif graph_type == "Horizon Chart":
        num_bands = 5
        pos_colors = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
        neg_colors = ["#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]

        # Safely compute absolute max data peak
        valid_z = z_data[~np.isnan(z_data)] if len(z_data) else []
        actual_max_abs = np.nanmax(np.abs(valid_z)) if len(valid_z) else 1
        if actual_max_abs == 0 or np.isnan(actual_max_abs): actual_max_abs = 1

        # Calculate nice symmetrical ticks bounded outward from the data
        _, padded_max_abs, pos_ticks, pos_labels = calculate_nice_ticks(0, actual_max_abs, 4)
        band_size = padded_max_abs / num_bands

        for i, name in enumerate(y_names):
            z = np.array(z_data[i], dtype=float)
            z_safe = np.nan_to_num(z, nan=0.0)
            custom_data_arr = [name] * len(x_data)

            # Invisible trace for capturing hover interactions per stock
            fig.add_trace(
                go.Scatter(x=x_data, y=[i + 0.5] * len(x_data), mode="lines", line=dict(color='rgba(0,0,0,0)', width=1),
                           name=name, customdata=custom_data_arr, cliponaxis=False,
                           text=[f"{val:.2f}" if not np.isnan(val) else "NaN" for val in z],
                           hovertemplate="<b>%{customdata}</b><br>Time: %{x}<br>Value: %{text}<extra></extra>",
                           hoverinfo="all",
                           showlegend=False))

            for b in range(num_bands):
                pos_vals = np.clip(z_safe - b * band_size, 0, band_size)
                fig.add_trace(
                    go.Scatter(x=x_data, y=[i] * len(x_data), mode="lines", line=dict(width=0), showlegend=False,
                               hoverinfo="skip", customdata=custom_data_arr, cliponaxis=False))
                fig.add_trace(go.Scatter(x=x_data, y=i + (pos_vals / band_size), mode="lines", fill="tonexty",
                                         fillcolor=pos_colors[b],
                                         line=dict(width=0),
                                         showlegend=False, hoverinfo="skip", customdata=custom_data_arr,
                                         cliponaxis=False))

                neg_vals = np.clip(-z_safe - b * band_size, 0, band_size)
                fig.add_trace(
                    go.Scatter(x=x_data, y=[i] * len(x_data), mode="lines", line=dict(width=0), showlegend=False,
                               hoverinfo="skip", customdata=custom_data_arr, cliponaxis=False))
                fig.add_trace(go.Scatter(x=x_data, y=i + (neg_vals / band_size), mode="lines", fill="tonexty",
                                         fillcolor=neg_colors[b],
                                         line=dict(width=0),
                                         showlegend=False, hoverinfo="skip", customdata=custom_data_arr,
                                         cliponaxis=False))

            # Clean grey separator line at the baseline
            fig.add_trace(go.Scatter(x=x_data, y=[i] * len(x_data), mode="lines", line=dict(width=1, color="#444444"),
                                     showlegend=False, hoverinfo="skip", customdata=custom_data_arr, cliponaxis=False))

        # Dummy trace for continuous (stepped) colorbar
        custom_colorscale = [
            [0.0, neg_colors[4]], [0.1, neg_colors[4]],
            [0.1, neg_colors[3]], [0.2, neg_colors[3]],
            [0.2, neg_colors[2]], [0.3, neg_colors[2]],
            [0.3, neg_colors[1]], [0.4, neg_colors[1]],
            [0.4, neg_colors[0]], [0.5, neg_colors[0]],
            [0.5, pos_colors[0]], [0.6, pos_colors[0]],
            [0.6, pos_colors[1]], [0.7, pos_colors[1]],
            [0.7, pos_colors[2]], [0.8, pos_colors[2]],
            [0.8, pos_colors[3]], [0.9, pos_colors[3]],
            [0.9, pos_colors[4]], [1.0, pos_colors[4]]
        ]

        # Combine symmetric negative and positive ticks
        tickvals = [-t for t in reversed(pos_ticks[1:])] + pos_ticks
        ticktext = [f"-{l}" if l != "0" else "0" for l in reversed(pos_labels[1:])] + pos_labels

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                colorscale=custom_colorscale,
                cmin=-padded_max_abs,
                cmax=padded_max_abs,
                showscale=True,
                colorbar=dict(
                    title=value_title,
                    thickness=15,
                    outlinewidth=0,
                    lenmode="pixels",
                    len=250,
                    yanchor="top",
                    y=1,
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext
                )
            ),
            showlegend=False, hoverinfo="none"
        ))

        fig.update_layout(title=title_text, height=fig_height,
                          xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
                          yaxis={"title": "Stock Ticker", "tickvals": [i + 0.5 for i in range(len(y_names))],
                                 "ticktext": y_names,
                                 "range": [-0.5, max(1, len(y_names))], "automargin": True},
                          clickmode="event+select",
                          hovermode="closest",
                          plot_bgcolor="#f9f9f9",
                          showlegend=False,
                          margin=dict(t=70, b=40, l=80, r=20))

    elif graph_type == "Spark Line":
        valid_z = z_data[~np.isnan(z_data)] if len(z_data) else []
        max_abs = np.nanmax(np.abs(valid_z)) if len(valid_z) else 1
        if max_abs == 0 or np.isnan(max_abs): max_abs = 1

        trace_index = 0
        data_line_indices = []

        for i, name in enumerate(y_names):
            z = np.array(z_data[i], dtype=float)
            custom_data_arr = [name] * len(x_data)

            # Separator line at baseline. Set color to #444444
            fig.add_trace(go.Scatter(x=x_data, y=[i] * len(x_data), mode="lines", line=dict(width=1, color="#444444"),
                                     showlegend=False, hoverinfo="skip", customdata=custom_data_arr, cliponaxis=False))
            trace_index += 1

            baseline_gaps = [i if not np.isnan(val) else None for val in z]
            valid_mask = ~np.isnan(z)

            if np.any(valid_mask):
                # Positive Fill Area
                if np.max(z[valid_mask]) > 0:
                    fig.add_trace(
                        go.Scatter(x=x_data, y=baseline_gaps, mode="lines", line=dict(width=0, color='rgba(0,0,0,0)'),
                                   showlegend=False, hoverinfo="skip", customdata=custom_data_arr, connectgaps=False,
                                   cliponaxis=False))
                    trace_index += 1
                    fig.add_trace(go.Scatter(x=x_data, y=i + (
                            np.where(np.isnan(z), np.nan, np.clip(z, 0, None)) / max_abs) * 0.45, mode="lines",
                                             fill="tonexty", fillcolor="rgba(31, 119, 180, 0.4)",
                                             line=dict(width=0), showlegend=False, hoverinfo="skip",
                                             customdata=custom_data_arr, connectgaps=False, cliponaxis=False))
                    trace_index += 1

                # Negative Fill Area
                if np.min(z[valid_mask]) < 0:
                    fig.add_trace(
                        go.Scatter(x=x_data, y=baseline_gaps, mode="lines", line=dict(width=0, color='rgba(0,0,0,0)'),
                                   showlegend=False, hoverinfo="skip", customdata=custom_data_arr, connectgaps=False,
                                   cliponaxis=False))
                    trace_index += 1
                    fig.add_trace(go.Scatter(x=x_data, y=i + (
                            np.where(np.isnan(z), np.nan, np.clip(z, None, 0)) / max_abs) * 0.45, mode="lines",
                                             fill="tonexty", fillcolor="rgba(214, 39, 40, 0.4)",
                                             line=dict(width=0), showlegend=False, hoverinfo="skip",
                                             customdata=custom_data_arr, connectgaps=False, cliponaxis=False))
                    trace_index += 1

                # Actual data tracking line with gradient marker overlay
                real_z = np.where(np.isnan(z), np.nan, z)
                text_vals = [f"{val:.2f}" if not np.isnan(val) else "NaN" for val in real_z]

                data_line_indices.append(trace_index)
                fig.add_trace(go.Scatter(
                    x=x_data, y=i + (real_z / max_abs) * 0.45,
                    mode="lines+markers",
                    line=dict(width=1.5, color="#bbbbbb"),  # Subtle neutral connecting line
                    marker=dict(
                        size=5,
                        color=real_z,  # This maps the value directly to the colorscale
                        colorscale=[[0, '#d62728'], [0.45, '#bbbbbb'], [0.55, '#bbbbbb'], [1, '#1f77b4']],
                        cmin=-max_abs,
                        cmax=max_abs,
                        showscale=False,
                        line=dict(width=0)
                    ),
                    name=name,
                    text=text_vals,
                    customdata=custom_data_arr,
                    hovertemplate="<b>%{customdata}</b><br>Time: %{x}<br>Value: %{text}<extra></extra>",
                    showlegend=False,
                    hoverinfo="all",
                    connectgaps=False,
                    cliponaxis=False
                ))
                trace_index += 1

        # Disable all legends for Spark Line, set closest hovermode, and add the native in-chart button control
        fig.update_layout(title=title_text, height=fig_height,
                          xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
                          yaxis={"title": "Stock Ticker", "tickvals": [i + 0.5 for i in range(len(y_names))],
                                 "ticktext": y_names,
                                 "range": [-0.5, max(1, len(y_names))], "automargin": True},
                          clickmode="event+select",
                          hovermode="closest",
                          plot_bgcolor="#f9f9f9",
                          showlegend=False,
                          margin=dict(t=70, b=40, l=80, r=20),
                          updatemenus=[
                              dict(
                                  type="buttons",
                                  direction="left",
                                  buttons=[
                                      dict(
                                          args=[{"mode": "lines"}, data_line_indices],
                                          label="Hide Markers",
                                          method="restyle"
                                      ),
                                      dict(
                                          args=[{"mode": "lines+markers"}, data_line_indices],
                                          label="Show Markers",
                                          method="restyle"
                                      )
                                  ],
                                  pad={"r": 10, "t": 10},
                                  showactive=True,
                                  x=1.0,
                                  xanchor="right",
                                  y=1.05,
                                  yanchor="bottom",
                                  bgcolor="#ffffff",
                                  bordercolor="#007bff",
                                  font=dict(size=12, color="#007bff")
                              )
                          ])

    else:  # Line Chart
        # vibrant_palette combines several distinct palettes while explicitly removing
        # the trailing dark grey/black/brown colors from Vivid, Prism, and Set1
        vibrant_palette = (
                                  px.colors.qualitative.Plotly +
                                  px.colors.qualitative.Vivid[:-1] +
                                  px.colors.qualitative.Prism[:-1] +
                                  px.colors.qualitative.Set1[:-2] +
                                  px.colors.qualitative.Set2 +
                                  px.colors.qualitative.Set3
                          ) * 10

        for i, name in enumerate(y_names):
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=z_data[i],
                    name=name,
                    mode="lines",
                    line=dict(color=vibrant_palette[i]),
                    customdata=[name] * len(x_data)
                )
            )
        fig.update_layout(title=title_text, height=fig_height,
                          xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]}, yaxis={"title": value_title},
                          showlegend=True,
                          legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02,
                                  "itemclick": False, "itemdoubleclick": False},
                          clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9")

    return fig


def main():
    global ticker_to_sector, sorted_sectors
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    all_data, names = load_all_data()

    if not all_data:
        print("Application stopped because no data was loaded.")
        return

    # --- DYNAMIC TIME EXTRACTION ---
    # Find the global minimum and maximum timestamps across all loaded stocks
    all_times = pd.concat([df['Time'] for df in all_data])

    # Convert to standard US Market Time (Eastern Time)
    min_time = pd.to_datetime(all_times.min(), unit='s').tz_localize('UTC').tz_convert('America/New_York')
    max_time = pd.to_datetime(all_times.max(), unit='s').tz_localize('UTC').tz_convert('America/New_York')

    # Format the string based on whether it's 1 day or multiple days
    if min_time.date() == max_time.date():
        date_str = min_time.strftime('%A, %B %d, %Y')
        time_range_str = f"{min_time.strftime('%I:%M %p')} - {max_time.strftime('%I:%M %p')} ET"
        dynamic_timeframe = f"{date_str} ({time_range_str})"
    else:
        dynamic_timeframe = f"{min_time.strftime('%B %d, %Y')} to {max_time.strftime('%B %d, %Y')}"
    # -------------------------------

    # Fetch Categories
    ticker_to_sector = fetch_sectors()
    for name in names:
        if name not in ticker_to_sector:
            ticker_to_sector[name] = 'Unknown'

    sorted_sectors = sorted(list(set(ticker_to_sector.values())))

    print(f"Loaded {len(all_data)} stocks across {len(sorted_sectors)} categories.")
    print(f"Detected Data Range: {dynamic_timeframe}")
    print("Aggregating base data (this may take a few seconds)...")

    # Initial graph placeholders to allow Dash callbacks to render the persisted state safely
    placeholder_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))
    placeholder_fig.update_layout(annotations=[
        {"text": "Select Stock using the dropdown above to view the price graph", "xref": "paper", "yref": "paper",
         "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 20}}])

    # We only register this ONCE when setting up the app.
    placeholder_fig.register_update_graph_callback(app=app, graph_id="price_graph")
    bottom_fig = go.Figure()  # Created empty; gets updated on page load automatically

    initial_stock_options = [{"label": n, "value": n} for n in names]

    # --- Context Header Block ---
    header_block = html.Div([
        html.H2("S&P 500 Visualization Dashboard", style={"margin": "0", "color": "#ffffff", "fontWeight": "bold"}),
        html.P(f"Asset: S&P 500 (Hera ETF) | Timeframe: {dynamic_timeframe}",
               style={"margin": "0", "marginTop": "5px", "color": "#d0d0d0", "fontSize": "0.95rem"})
    ], style={"backgroundColor": "#343a40", "padding": "1.2rem", "borderRadius": "10px", "marginBottom": "1rem",
              "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"})

    app.layout = html.Div([

        header_block,

        html.Div([
            html.Div([
                html.P("Select Stock (Top Graph):", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="ticker_selector",
                    options=initial_stock_options,
                    value=names[0] if names else None,
                    clearable=False,
                    persistence=True,
                    persistence_type='local'
                ),
            ], style={"marginBottom": "1rem", "padding": "0.5rem", "backgroundColor": "#ffffff", "borderRadius": "10px",
                      "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),
            dcc.Loading(dcc.Graph(id="price_graph", figure=placeholder_fig, config={
                "toImageButtonOptions": {"format": "png", "filename": "price_graph", "width": 1920, "height": 1080,
                                         "scale": 3}}), type="circle")
        ], style={"marginBottom": "1rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "borderRadius": "10px",
                  "padding": "0.5rem", "backgroundColor": "#f9f9f9"}),

        html.Div([
            html.Div([
                html.P("Select a metric:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(id="metric_dropdown", options=[{"label": "Stock Price", "value": "Price"}], value=metric,
                             clearable=False, persistence=True, persistence_type='local',
                             style={"marginBottom": "0.5rem"}),
                html.P(id="metric_description", style={"marginBottom": "1rem", "fontStyle": "italic"}),

                html.P("Select an aggregation function:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(id="aggregation_dropdown",
                             options=[{"label": x, "value": x} for x in ["Mean", "Median", "Max", "Min", "Std"]],
                             value=chosen_aggregation, clearable=False, persistence=True, persistence_type='local',
                             style={"marginBottom": "1rem"}),

                html.P("Select a time window (minutes):", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Input(id="time_window_input", type="number", value=time_window_aggregation_minutes, min=1, max=720,
                          step=1, placeholder="Time Window", persistence=True, persistence_type='local',
                          style={"width": "100%", "marginBottom": "1rem"}),

                html.P("Data Processing:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(id="data_processing_dropdown", options=[{"label": "None (Raw Values)", "value": "none"},
                                                                     {"label": "Normalize (Min-Max)",
                                                                      "value": "normalize"},
                                                                     {"label": "Percentage Change (%)",
                                                                      "value": "pct_change"}], value="none",
                             clearable=False, persistence=True, persistence_type='local',
                             style={"marginBottom": "1rem"}),

                html.P("Select bottom graph style:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(id="graph_type_dropdown", options=[{"label": " Heatmap", "value": "Heatmap"},
                                                                {"label": " 2D Line chart", "value": "Line Chart"},
                                                                {"label": " 3D Line chart", "value": "3D Lines"},
                                                                {"label": " Horizon Chart", "value": "Horizon Chart"},
                                                                {"label": " Spark Line", "value": "Spark Line"},
                                                                {"label": " Correlation Matrix",
                                                                 "value": "Correlation Matrix"},
                                                                {"label": " UMAP Clusters", "value": "UMAP Clusters"},
                                                                {"label": " t-SNE Clusters", "value": "t-SNE Clusters"}],
                             value="Heatmap", clearable=False, persistence=True, persistence_type='local',
                             style={"marginBottom": "1rem"}),

                html.P("Sort Stocks By:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="sort_by_dropdown",
                    options=[
                        {"label": "Default (Original Order)", "value": "Default"},
                        {"label": "Alphabetical (A-Z)", "value": "Alphabetical (A-Z)"},
                        {"label": "Alphabetical (Z-A)", "value": "Alphabetical (Z-A)"},
                        {"label": "Correlation (Group Similar)", "value": "Correlation"}
                    ],
                    value="Default",
                    clearable=False,
                    persistence=True,
                    persistence_type='local',
                    style={"marginBottom": "1rem"}
                ),

                html.P("Filter by Category (Sector):",
                       style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="sector_selector",
                    options=[{"label": "All Categories", "value": "ALL"}] + [{"label": s, "value": s} for s in
                                                                             sorted_sectors],
                    value="ALL",
                    clearable=False,
                    persistence=True,
                    persistence_type='local',
                    style={"marginBottom": "1.5rem"}
                ),

                html.P("Filter Specific Stocks:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="timeseries_selector",
                    options=initial_stock_options,
                    value=names.copy(),
                    multi=True,
                    placeholder="Select stocks...",
                    persistence=True,
                    persistence_type='local',
                    style={"marginBottom": "0.5rem"}
                ),
                html.Div([
                    html.Button("Select All", id="select_all_btn", n_clicks=0,
                                style={"width": "48%", "marginRight": "4%", "padding": "0.4rem",
                                       "backgroundColor": "#007bff", "color": "white", "border": "none",
                                       "borderRadius": "6px"}),
                    html.Button("Unselect All", id="unselect_all_btn", n_clicks=0,
                                style={"width": "48%", "padding": "0.4rem", "backgroundColor": "#dc3545",
                                       "color": "white", "border": "none", "borderRadius": "6px"}),
                ], style={"display": "flex", "marginBottom": "1rem"}),

                html.Button("Apply", id="update_bottom_graph_button", n_clicks=0,
                            style={"marginTop": "1rem", "width": "100%", "padding": "0.75rem 1rem", "fontSize": "1rem",
                                   "fontWeight": "bold", "color": "#fff", "backgroundColor": "#28a745",
                                   "border": "none", "borderRadius": "8px",
                                   "boxShadow": "0 4px 6px rgba(40, 167, 69, 0.3)", "cursor": "pointer"}),
            ], style={"flex": "0 0 300px", "padding": "0.5rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                      "borderRadius": "10px", "backgroundColor": "#ffffff", "minWidth": "250px", "position": "sticky",
                      "top": "1rem"}),

            html.Div([
                dcc.Loading(
                    html.Div([
                        # --- t-SNE Parameters Overlay ---
                        html.Div([
                            html.P("t-SNE Perplexity:",
                                   style={"fontWeight": "bold", "marginBottom": "0.2rem", "fontSize": "0.9rem"}),
                            dcc.Slider(id="tsne_perplexity", min=2, max=50, step=1, value=30, marks={2: '2', 50: '50'},
                                       tooltip={"placement": "bottom", "always_visible": True}, updatemode='mouseup'),
                        ], id="tsne_params",
                            style={"display": "none", "position": "absolute", "top": "70px", "left": "15px",
                                   "width": "280px", "zIndex": "1000", "background": "rgba(255,255,255,0.9)",
                                   "padding": "10px", "borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0,0,0,0.2)",
                                   "border": "1px solid #dee2e6"}),

                        # --- UMAP Parameters Overlay ---
                        html.Div([
                            html.P("UMAP Neighbors:",
                                   style={"fontWeight": "bold", "marginBottom": "0.2rem", "fontSize": "0.9rem"}),
                            dcc.Slider(id="umap_neighbors", min=2, max=50, step=1, value=15, marks={2: '2', 50: '50'},
                                       tooltip={"placement": "bottom", "always_visible": True}, updatemode='mouseup'),
                            html.P("UMAP Min Distance:",
                                   style={"fontWeight": "bold", "marginBottom": "0.2rem", "marginTop": "0.8rem",
                                          "fontSize": "0.9rem"}),
                            dcc.Slider(id="umap_min_dist", min=0.0, max=0.99, step=0.05, value=0.1,
                                       marks={0: '0', 1: '1'}, tooltip={"placement": "bottom", "always_visible": True},
                                       updatemode='mouseup'),
                        ], id="umap_params",
                            style={"display": "none", "position": "absolute", "top": "70px", "left": "15px",
                                   "width": "280px", "zIndex": "1000", "background": "rgba(255,255,255,0.9)",
                                   "padding": "10px", "borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0,0,0,0.2)",
                                   "border": "1px solid #dee2e6"}),

                        # --- The Graph ---
                        dcc.Graph(id="bottom_graph", figure=bottom_fig, clear_on_unhover=True, config={
                            "toImageButtonOptions": {"format": "png", "filename": "bottom_graph", "width": 1920,
                                                     "height": 1080, "scale": 3}
                        })
                    ], style={"position": "relative", "width": "100%", "height": "100%"})
                    , type="circle")
            ], style={"flex": "1", "marginLeft": "1rem", "padding": "0.5rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                      "borderRadius": "10px", "backgroundColor": "#f9f9f9", "minWidth": "0", "maxHeight": "100vh",
                      "overflowY": "auto"})
        ], style={"display": "flex", "flexWrap": "nowrap", "alignItems": "flex-start", "gap": "1rem"}),

        # --- DUMMY OUTPUT REQUIRED FOR CLIENTSIDE JAVASCRIPT CALLBACK ---
        html.Div(id="dummy-clientside-output", style={"display": "none"})

    ], style={"padding": "0.5rem", "fontFamily": "Arial, sans-serif", "backgroundColor": "#f0f2f5"})

    # --- CALLBACKS ---

    @app.callback(
        [Output("tsne_params", "style"), Output("umap_params", "style")],
        Input("graph_type_dropdown", "value"),
        State("tsne_params", "style"),
        State("umap_params", "style")
    )
    def toggle_cluster_params(gtype, current_tsne_style, current_umap_style):
        tsne_style = current_tsne_style.copy() if current_tsne_style else {}
        tsne_style["display"] = "block" if gtype == "t-SNE Clusters" else "none"

        umap_style = current_umap_style.copy() if current_umap_style else {}
        umap_style["display"] = "block" if gtype == "UMAP Clusters" else "none"

        return tsne_style, umap_style

    @app.callback(
        Output("timeseries_selector", "options"),
        Output("timeseries_selector", "value"),
        Input("sector_selector", "value"),
        Input("select_all_btn", "n_clicks"),
        Input("unselect_all_btn", "n_clicks"),
        State("timeseries_selector", "value")
    )
    def update_stock_dropdown(selected_sector, select_all, unselect_all, current_selection):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        if selected_sector == "ALL":
            available_stocks = names
        else:
            available_stocks = [n for n in names if ticker_to_sector.get(n) == selected_sector]

        options = [{"label": n, "value": n} for n in available_stocks]

        if trigger == "unselect_all_btn":
            return options, []
        elif trigger == "select_all_btn" or trigger == "sector_selector":
            # Automatically select all available stocks when changing sectors
            return options, available_stocks
        else:
            # Handles initial load
            if current_selection is not None:
                valid_selection = [s for s in current_selection if s in available_stocks]
                return options, valid_selection

            return options, available_stocks

    @app.callback(Output("metric_description", "children"), Input("metric_dropdown", "value"))
    def update_description(selected_metric):
        return metric_descriptions_map[selected_metric] if selected_metric in metric_descriptions_map else ""

    @app.callback(Output("price_graph", "figure"), Input("ticker_selector", "value"))
    def update_price_graph_from_dropdown(selected_name):
        if not selected_name: return dash.no_update
        idx = names.index(selected_name)
        data = all_data[idx]
        fig = create_price_graph(data["Time"].values, data["Price"].values, selected_name)
        return fig

    @app.callback(
        Output("bottom_graph", "figure"),
        Input("update_bottom_graph_button", "n_clicks"),
        Input("graph_type_dropdown", "value"),
        Input("data_processing_dropdown", "value"),
        Input("timeseries_selector", "value"),
        Input("sort_by_dropdown", "value"),
        Input("tsne_perplexity", "value"),
        Input("umap_neighbors", "value"),
        Input("umap_min_dist", "value"),
        State("metric_dropdown", "value"),
        State("aggregation_dropdown", "value"),
        State("time_window_input", "value"),
        State("bottom_graph", "figure")
    )
    def update_bottom_graph(btn_clicks, selected_graph_type, data_processing, selected_timeseries, sort_by,
                            tsne_perp, umap_neigh, umap_dist,
                            selected_metric, selected_aggregation, selected_time_window, bottom_fig):
        global last_update_bottom_graph_click_count, last_graph_type, last_data_processing_state, last_sort_by_state
        updated = False

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ""

        # Only block updates if a non-relevant event fired
        if trigger_id not in ["update_bottom_graph_button", "graph_type_dropdown", "data_processing_dropdown",
                              "timeseries_selector", "sort_by_dropdown", "tsne_perplexity", "umap_neighbors",
                              "umap_min_dist", ""]:
            return dash.no_update

        # Avoid redundant rendering if button is clicked but no states have actually changed
        if trigger_id == "update_bottom_graph_button" and last_update_bottom_graph_click_count == btn_clicks and selected_graph_type == last_graph_type and data_processing == last_data_processing_state and sort_by == last_sort_by_state:
            return dash.no_update

        last_update_bottom_graph_click_count = btn_clicks
        last_graph_type = selected_graph_type
        last_data_processing_state = data_processing
        last_sort_by_state = sort_by

        selected_time_window_sec = selected_time_window * MINUTE_SEC
        new_z, new_x = aggregate_data(all_data, metric=selected_metric,
                                      aggregation=aggregation_functions_map[selected_aggregation],
                                      time_window_sec=selected_time_window_sec)

        if data_processing == "normalize":
            new_z = normalize_data(new_z)
        elif data_processing == "pct_change":
            new_z = pct_change_data(new_z)

        if selected_timeseries:
            selected_indices = [i for i, n in enumerate(names) if n in selected_timeseries]
            filtered_names = [names[i] for i in selected_indices]
            filtered_z = np.array(new_z)[selected_indices]

            # --- APPLIED SORTING ALGORITHMS ---
            if sort_by == "Alphabetical (A-Z)":
                sort_idx = np.argsort(filtered_names)
                filtered_names = [filtered_names[i] for i in sort_idx]
                filtered_z = filtered_z[sort_idx]
            elif sort_by == "Alphabetical (Z-A)":
                sort_idx = np.argsort(filtered_names)[::-1]
                filtered_names = [filtered_names[i] for i in sort_idx]
                filtered_z = filtered_z[sort_idx]
            elif sort_by == "Correlation" and len(filtered_z) > 1:
                # Adding microscopic random noise prevents division-by-zero NaNs in perfectly flat data segments
                z_filled = np.nan_to_num(filtered_z, nan=0.0) + np.random.normal(0, 1e-9, filtered_z.shape)

                # Greedy path sorting algorithm (Finds closest matching neighbor sequentially)
                corr_matrix = np.corrcoef(z_filled)
                unvisited = set(range(len(filtered_names)))

                start_idx = 0
                sort_idx = [start_idx]
                unvisited.remove(start_idx)
                curr_idx = start_idx

                while unvisited:
                    # Find the index with the highest correlation to the current index
                    next_idx = max(unvisited, key=lambda i: corr_matrix[curr_idx, i])
                    sort_idx.append(next_idx)
                    unvisited.remove(next_idx)
                    curr_idx = next_idx

                filtered_names = [filtered_names[i] for i in sort_idx]
                filtered_z = filtered_z[sort_idx]
        else:
            filtered_names = []
            filtered_z = np.array([])

        current_range = None
        if bottom_fig and isinstance(bottom_fig, dict) and "layout" in bottom_fig and "xaxis" in bottom_fig[
            "layout"] and "range" in bottom_fig["layout"]["xaxis"]:
            current_range = bottom_fig["layout"]["xaxis"]["range"]
        elif bottom_fig and hasattr(bottom_fig, "layout") and hasattr(bottom_fig.layout, "xaxis") and hasattr(
                bottom_fig.layout.xaxis, "range"):
            current_range = bottom_fig.layout.xaxis.range

        # Pass clean names (no categories) to the bottom figure
        bottom_fig = create_bottom_figure(filtered_z, new_x, filtered_names, selected_graph_type,
                                          selected_metric, selected_aggregation, data_processing,
                                          tsne_perp, umap_neigh, umap_dist)

        # DO NOT preserve the saved zoom level if the user explicitly switches the graph type.
        # Doing so breaks the layout mapping of the new chart type causing the weird visualization bug.
        if trigger_id not in ["update_bottom_graph_button", "graph_type_dropdown"] and current_range:
            if selected_graph_type == "3D Lines":
                bottom_fig.update_layout(scene=dict(xaxis={"range": current_range}))
            elif selected_graph_type not in ["Correlation Matrix", "UMAP Clusters", "t-SNE Clusters"]:
                bottom_fig.update_layout(xaxis={"range": current_range})
        updated = True

        return bottom_fig if updated else dash.no_update

    # --- CLIENTSIDE CALLBACK FOR MULTI-SELECT LEGEND HIGHLIGHTING ---
    app.clientside_callback(
        """
        function(figure) {
            setTimeout(function() {
                const graph = document.getElementById('bottom_graph');
                if (!graph) return;
                const plot = graph.querySelector('.js-plotly-plot');
                if (!plot) return;

                // Clear state on new graph initialization
                plot._selectedTraces = new Set();
                plot._originalColors = null;

                function updateStyles(hoveredIndex = -1) {
                    const traceCount = plot.data.length;
                    const isCluster = traceCount > 0 && (plot.data[0].mode === 'markers+text' || plot.data[0].mode === 'markers');

                    // Capture original colors strictly once per full dataset render to avoid restyle mutations
                    if (!isCluster && !plot._originalColors) {
                        plot._originalColors = plot._fullData.map(t => (t.line || {}).color || '#000000');
                    }

                    let update = { opacity: [] };
                    if (!isCluster) update.line = [];

                    const hasSelection = plot._selectedTraces.size > 0;
                    const anyActive = hasSelection || hoveredIndex !== -1;

                    for (let i = 0; i < traceCount; i++) {
                        const isActive = plot._selectedTraces.has(i) || i === hoveredIndex;
                        const is3D = !isCluster && plot.data[i].type === 'scatter3d';

                        if (!anyActive) {
                            // Reset state (No selections, no hover)
                            update.opacity.push(1);
                            if (!isCluster) {
                                let origWidth = is3D ? 4 : 2;
                                update.line.push({ width: origWidth, color: plot._originalColors[i] });
                            }
                        } else if (isActive) {
                            // Highlighted focus state
                            update.opacity.push(1);
                            if (!isCluster) {
                                let activeWidth = is3D ? 8 : 5;
                                update.line.push({ width: activeWidth, color: plot._originalColors[i] });
                            }
                        } else {
                            // Defocused greyed-out state 
                            // Opacita pro 3D je vyšší (0.4) aby čáry lépe vynikly, pro 2D stačí 0.25
                            update.opacity.push(is3D ? 0.4 : 0.25);
                            if (!isCluster) {
                                // Tlustší neaktivní čára ve 3D a tmavší odstín šedé
                                let unselectedWidth = is3D ? 2.5 : 1.5;
                                update.line.push({ width: unselectedWidth, color: "rgba(130,130,130,0.9)" });
                            }
                        }
                    }
                    Plotly.restyle(plot, update);
                }

                function attachLegendEvents() {
                    const legendItems = plot.querySelectorAll('.legend .traces');
                    if (!legendItems.length) return;

                    legendItems.forEach((item, index) => {
                        // Prevent multi-bindings if Plotly rerenders legend
                        item.onmouseenter = null;
                        item.onmouseleave = null;
                        item.onclick = null;

                        item.onmouseenter = function() {
                            updateStyles(index);
                        };

                        item.onmouseleave = function() {
                            updateStyles(-1);
                        };

                        item.onclick = function(e) {
                            e.stopPropagation(); // Block default behavior passing through the DOM
                            e.preventDefault();

                            // Toggle Set presence
                            if (plot._selectedTraces.has(index)) {
                                plot._selectedTraces.delete(index);
                            } else {
                                plot._selectedTraces.add(index);
                            }

                            // Re-render styles considering we are still actively hovering after the click
                            updateStyles(index);
                        };
                    });
                }

                attachLegendEvents();
                if (!plot._legendEventsAttached) {
                    plot.on('plotly_afterplot', attachLegendEvents);
                    plot._legendEventsAttached = true;
                }
            }, 300);
            return window.dash_clientside.no_update;
        }
        """,
        Output("dummy-clientside-output", "children"),
        Input("bottom_graph", "figure"),
        prevent_initial_call=False
    )

    print(f"Starting server at http://{HOST_ADDRESS}:{PORT}")
    app.run(host=HOST_ADDRESS, port=PORT, debug=False)


if __name__ == "__main__":
    print("Starting...")
    main()