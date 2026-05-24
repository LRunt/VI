import os
import sys
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
    print(
        "Warning: 'umap-learn' is not installed. UMAP clustering will be disabled. Run 'pip install umap-learn' to enable.")

# --- Optional import for t-SNE ---
try:
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print(
        "Warning: 'scikit-learn' is not installed. t-SNE clustering will be disabled. Run 'pip install scikit-learn' to enable.")

# --- Data Loading Logic ---

MINUTE_SEC = 60
HOUR_SEC = 60 * MINUTE_SEC
DAY_SEC = 24 * HOUR_SEC

# Global variables for categorization (keyed by LOB name -> category label)
lob_to_category = {}
sorted_categories = []


def load_data(date, market_segment_id, security, level_depth=1):
    lobster_fp = f"data/{date}_{market_segment_id}_{security}_lobster_augmented.csv"
    data = pd.read_csv(lobster_fp, sep=",")

    i = level_depth + 1
    while True:
        if f"Ask Price {i}" not in data.columns and f"Bid Price {i}" not in data.columns:
            break
        data = data.drop(columns=[f"Ask Price {i}", f"Bid Price {i}"], errors="ignore")
        i += 1

    return data


def load_ensemble_results(date, market_segment_id, security):
    res_dir = f"res/{date}_{market_segment_id}_{security}"
    if not os.path.exists(res_dir):
        return None
    files = os.listdir(res_dir)
    if not files:
        return None

    temp_results = []
    for file in files:
        if file.endswith(".json") and ("if_" in file or "ffnn_" in file or "cnn_" in file or "transformer_" in file):
            with open(os.path.join(res_dir, file), "r") as fp:
                store = json.load(fp)
                y_scores = np.array(store["y_scores"])
                if "if_" in file:
                    y_scores = -y_scores
                y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
                temp_results.append(y_scores)

    if not temp_results:
        return None

    majority_len = np.median([len(x) for x in temp_results])
    temp_results = [x[:int(majority_len)] for x in temp_results]
    y_scores_ensemble = np.mean(temp_results, axis=0)

    def transform_ys(y_scores, contamination=0.01, lower_is_better=True):
        how_many_can_be = len(y_scores) * (1 - contamination)
        y_pred = np.zeros_like(y_scores)
        if lower_is_better:
            y_pred[np.argsort(y_scores)[:int(how_many_can_be)]] = 1
            y_pred[np.argsort(y_scores)[int(how_many_can_be):]] = -1
        else:
            y_pred[np.argsort(y_scores)[::-1][:int(how_many_can_be)]] = 1
            y_pred[np.argsort(y_scores)[::-1][int(how_many_can_be):]] = -1
        y_scores_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        anomaly_proba = y_scores_norm if lower_is_better else 1 - y_scores_norm
        return y_pred, anomaly_proba

    y_pred_ensemble, anomaly_proba_ensemble = transform_ys(y_scores_ensemble, contamination=0.01, lower_is_better=True)
    threshold = np.percentile(y_scores_ensemble, 99.9)
    y_pred_ensemble[anomaly_proba_ensemble < threshold] = 1
    anomaly_proba_ensemble[anomaly_proba_ensemble < threshold] = 0

    return y_pred_ensemble, anomaly_proba_ensemble


def load_all_data(level_depth=1):
    data = []
    names = []
    detections = []

    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' folder. Please place your CSV files there.")

    for file in os.listdir("data"):
        if file.endswith("_lobster_augmented.csv"):
            date, market_segment_id, security = file.split("_")[:3]
            names.append(f"{date}_{market_segment_id}_{security}")
            data.append(load_data(date, market_segment_id, security, level_depth=level_depth))
            detections.append(load_ensemble_results(date, market_segment_id, security))

    return data, names, detections


def fetch_categories(filename="lob_categories.json"):
    """Loads a name-to-category mapping from a local JSON file (optional)."""
    try:
        if os.path.exists(filename):
            print(f"Loading categories from '{filename}'...")
            with open(filename, "r") as f:
                return json.load(f)
        else:
            print(f"Info: '{filename}' not found. All entries will be grouped as 'Unknown'.")
            return {}
    except Exception as e:
        print(f"Warning: Could not read '{filename}'. Defaulting to 'Unknown'. ({e})")
        return {}


def aggregate_data(all_data, metric="Ask Price 1", aggregation=np.mean, time_window=3600):
    aggregated_data = []
    for data in all_data:
        tmp_agg = []
        timestamps = pd.to_datetime(data["Time"].values, unit="ns")
        timestamps_series = pd.Series(timestamps)
        seconds_since_midnight = (timestamps_series - timestamps_series.dt.normalize()).dt.total_seconds()

        for i in range(0, DAY_SEC, time_window):
            data_in_window = data[(seconds_since_midnight >= i) & (seconds_since_midnight < i + time_window)]
            data_in_window = data_in_window[metric].dropna()
            tmp_agg.append(np.nan if data_in_window.empty else aggregation(data_in_window.values))

        aggregated_data.append(np.array(tmp_agg))

    return np.array(aggregated_data)


# --- Helpers & Dashboard Logic ---

HOST_ADDRESS = "127.0.0.1"
PORT = 8080

timestamps_graph_labels = None
chosen_aggregation = "Mean"
aggregation_functions_map = {
    "Mean": np.mean, "Median": np.median, "Max": np.max, "Min": np.min, "Std": np.std
}
metric = "Ask Price 1"
metric_descriptions_map = {
    "Ask Price 1": "The lowest price a seller is willing to accept.",
    "Bid Price 1": "The highest price a buyer is willing to pay.",
    "Ask Volume 1": "The total number of offers available at the best ask price.",
    "Bid Volume 1": "The total number of offers available at the best bid price.",
    "Imbalance Index": "Measures the difference between buy and sell interest.",
    "Frequency of Incoming Messages": "Moving average (5 minutes window) of how often order book updates are received.",
    "Cancellations Rate": "Moving average (5 minutes window) of the rate at which orders are canceled.",
    "High Quoting Activity": "Indicates rapid updates in order quotes (changes in volumes).",
    "Unbalanced Quoting": "Shows bias towards one side of the market (buy or sell).",
    "Low Execution Probability": "\"Chance\" of execution given the current quoting activity.",
    "Trades Oppose Quotes": "Binary. 1 if the trade is on the opposite side of the more recently quoted side, 0 otherwise.",
    "Cancels Oppose Trades": "Binary. 1 if the trade is on the opposite side of the more recently canceled side, 0 otherwise."
}
time_window_aggregation = 60


def safe_corr(a, b):
    """Safely calculates correlation coefficient preventing zero-variance NaN issues."""
    a_safe = np.nan_to_num(a, nan=0.0)
    b_safe = np.nan_to_num(b, nan=0.0)

    if np.allclose(a_safe, b_safe, atol=1e-8):
        return 1.0

    std_a = np.std(a_safe)
    std_b = np.std(b_safe)

    if std_a == 0 or std_b == 0:
        return 0.0

    corr = np.corrcoef(a_safe, b_safe)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def filter_by_correlation(filtered_names, filtered_z, corr_ref_stock, corr_range):
    """
    Shared correlation filtering logic used by:
    - preview label
    - actual graph filtering

    Returns:
        kept_names
        kept_z
        valid_count
    """
    if (
            corr_ref_stock is None or
            corr_ref_stock == "NONE" or
            corr_ref_stock not in filtered_names
    ):
        return filtered_names, filtered_z, len(filtered_names)

    ref_local_idx = filtered_names.index(corr_ref_stock)
    ref_z = np.array(filtered_z[ref_local_idx], dtype=float)

    # Reference invalid
    if np.all(np.isnan(ref_z)):
        return [], np.array([]), 0

    ref_z = np.nan_to_num(ref_z, nan=0.0)

    # Reference constant
    if np.std(ref_z) == 0:
        return [], np.array([]), 0

    c_min, c_max = corr_range
    kept_names = []
    kept_z = []
    valid_count = 0

    for i, name in enumerate(filtered_names):
        target_z = np.array(filtered_z[i], dtype=float)

        # Skip all-NaN
        if np.all(np.isnan(target_z)):
            continue

        target_z = np.nan_to_num(target_z, nan=0.0)

        # Skip constant series
        if np.std(target_z) == 0:
            continue

        valid_count += 1

        # Always keep reference
        if name == corr_ref_stock:
            kept_names.append(name)
            kept_z.append(filtered_z[i])
            continue

        corr = np.corrcoef(ref_z, target_z)[0, 1]

        # Skip invalid corr
        if np.isnan(corr):
            continue

        if c_min <= corr <= c_max:
            kept_names.append(name)
            kept_z.append(filtered_z[i])

    return kept_names, np.array(kept_z), valid_count


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
            pct_arr = np.zeros_like(arr) if base_val == 0 else ((arr - base_val) / np.abs(base_val)) * 100
        else:
            pct_arr = np.zeros_like(arr)
        pct_z.append(pct_arr)
    return pct_z


def calculate_nice_ticks(vmin, vmax, target_ticks=5):
    if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
        return vmin, vmax, [vmin], [f"{vmin:.2f}"]

    span = vmax - vmin
    rough_step = span / (target_ticks - 1)
    mag = 10 ** math.floor(math.log10(rough_step)) if rough_step > 0 else 1
    rel_step = rough_step / mag

    if rel_step < 1.5:
        nice_step = 1 * mag
    elif rel_step < 3.5:
        nice_step = 2 * mag
    elif rel_step < 7.5:
        nice_step = 5 * mag
    else:
        nice_step = 10 * mag

    padded_min = math.floor(vmin / nice_step) * nice_step
    padded_max = math.ceil(vmax / nice_step) * nice_step
    ticks = np.arange(padded_min, padded_max + nice_step * 0.1, nice_step).tolist()

    def format_tick(val):
        val = round(val, 6)
        if float(val).is_integer():
            return str(int(val))
        elif nice_step >= 0.1:
            return f"{val:.1f}"
        elif nice_step >= 0.01:
            return f"{val:.2f}"
        else:
            return f"{val:.4f}"

    labels = [format_tick(t) for t in ticks]
    return padded_min, padded_max, ticks, labels


def create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels, name,
                       detected_anomalies, how_many_x_ticks=75):
    global timestamps_graph_labels
    timestamps_graph_labels = [
        datetime.datetime.fromtimestamp(int(ts) / 1e9 - HOUR_SEC).strftime("%H:%M:%S.%f")
        for ts in timestamps
    ]
    timestamps_graph = list(range(len(timestamps_graph_labels)))
    tickvals = list(range(0, len(timestamps), max(1, len(timestamps) // how_many_x_ticks)))
    ticklabels = [timestamps_graph_labels[i][:8] for i in tickvals]

    def interpolate_color(color1, color2, factor):
        return tuple(int(color1[i] + factor * (color2[i] - color1[i])) for i in range(3))

    price_graph_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))

    for i, ask_price in enumerate(ask_prices, 1):
        price_graph_fig.add_trace(
            go.Scattergl(name=f"Ask {i}", yaxis="y1"),
            hf_x=timestamps_graph, hf_y=ask_price,
            hf_marker_color=f"rgb" + str(interpolate_color((230, 31, 7), (255, 255, 255), (i - 1) / len(ask_prices)))
        )

    for i, bid_price in enumerate(bid_prices, 1):
        price_graph_fig.add_trace(
            go.Scattergl(name=f"Bid {i}", yaxis="y1"),
            hf_x=timestamps_graph, hf_y=bid_price,
            hf_marker_color=f"rgb" + str(interpolate_color((94, 163, 54), (255, 255, 255), (i - 1) / len(bid_prices)))
        )

    price_graph_fig.add_trace(
        go.Scattergl(name="Imbalance index", yaxis="y2", opacity=0.1),
        hf_x=timestamps_graph, hf_y=imbalance_indices, hf_marker_color="rgb(0, 0, 255)"
    )
    price_graph_fig.add_trace(
        go.Scattergl(name="Incoming messages (per sec)", yaxis="y3", opacity=0.25),
        hf_x=timestamps_graph, hf_y=freqs, hf_marker_color="rgb(255, 0, 215)"
    )
    price_graph_fig.add_trace(
        go.Scattergl(name="Cancellations rate", yaxis="y4", opacity=0.25),
        hf_x=timestamps_graph, hf_y=cancels, hf_marker_color="rgb(255, 215, 0)"
    )

    if detected_anomalies is not None:
        y_pred, anomaly_proba = detected_anomalies
        y_pred = np.pad(y_pred, (0, len(timestamps) - len(y_pred)), "constant", constant_values=1)[:len(timestamps)]
        anomaly_timestamps = [timestamps_graph[i] for i in range(len(y_pred)) if y_pred[i] == -1]

        prices = np.array(bid_prices + ask_prices)
        y_min, y_max = np.nanmin(prices), np.nanmax(prices)

        x_vals_with_Nones, y_vals_with_Nones = [], []
        for ts in anomaly_timestamps:
            x_vals_with_Nones.extend([ts, ts, None])
            y_vals_with_Nones.extend([y_min, y_max, None])

        price_graph_fig.add_trace(
            go.Scattergl(
                name="Detected Anomaly", yaxis="y1", mode="lines",
                line=dict(width=1, color="rgba(0, 0, 0, 0.75)"), hoverinfo="skip",
            ),
            hf_x=x_vals_with_Nones, hf_y=y_vals_with_Nones,
        )

    price_graph_fig.add_trace(
        go.Scattergl(
            name="Highlight", yaxis="y1", mode="lines", fill="toself",
            line=dict(width=2, color="rgba(25, 25, 100, 1)"),
            fillcolor="rgba(185, 215, 255, 0.3)", hoverinfo="skip", showlegend=False,
        ),
        hf_x=[], hf_y=[],
    )

    price_graph_fig.update_layout(
        title=f"{name}",
        xaxis={"title": "Timestamp", "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels,
               "range": [0, len(timestamps_graph_labels) - 1]},
        yaxis={"title": "Price", "side": "left"},
        yaxis2={"title": "Imbalance", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True,
                "range": [-1, 1]},
        yaxis3={"title": "Msgs/sec", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True},
        yaxis4={"title": "Cancels", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True},
        legend={"orientation": "h", "yanchor": "top", "y": -0.5, "xanchor": "center", "x": 0.5},
        clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9", margin=dict(b=100)
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
        value_title = "Value"

    num_series = max(1, len(y_names))
    if graph_type == "Heatmap":
        fig_height = max(400, num_series * 20 + 150)
    elif graph_type in ["Horizon Chart", "Spark Line"]:
        fig_height = max(500, num_series * 35 + 150)
    else:
        fig_height = 700

    if graph_type == "Heatmap":
        valid_z = np.array(z_data, dtype=float).flatten()
        valid_z = valid_z[~np.isnan(valid_z)]
        z_min = np.nanmin(valid_z) if len(valid_z) else 0
        z_max = np.nanmax(valid_z) if len(valid_z) else 1

        padded_min, padded_max, tickvals, ticktext = calculate_nice_ticks(z_min, z_max, target_ticks=5)

        fig.add_trace(go.Heatmap(
            z=z_data, x=x_data, y=y_names,
            colorscale="Viridis", zmin=padded_min, zmax=padded_max,
            colorbar=dict(title=value_title, thickness=15, outlinewidth=0, lenmode="pixels", len=250,
                          yanchor="top", y=1, tickmode="array", tickvals=tickvals, ticktext=ticktext),
            hoverongaps=False
        ))
        fig.update_layout(
            title=title_text, height=fig_height,
            xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
            yaxis={"title": "Day/Product", "range": [-0.5, max(1, len(y_names)) - 0.5]},
            clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9"
        )

    elif graph_type == "Correlation Matrix":
        n = len(y_names)
        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:
                    val = safe_corr(z_data[i], z_data[j])
                    corr_matrix[i, j] = val
                    corr_matrix[j, i] = val

        fig.add_trace(go.Heatmap(
            z=corr_matrix, x=y_names, y=y_names,
            colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
            colorbar=dict(title="Correlation", tickmode="array", tickvals=[-1, -0.5, 0, 0.5, 1],
                          ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]),
            hoverongaps=False,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        ))
        fig.update_layout(
            title=f"Correlation Matrix of {metric_name} ({agg_name})", height=800,
            xaxis={"title": "Day/Product", "tickangle": -45},
            yaxis={"title": "Day/Product", "autorange": "reversed"},
            plot_bgcolor="#f9f9f9", margin=dict(l=80, b=80)
        )

    elif graph_type == "UMAP Clusters":
        if not UMAP_AVAILABLE:
            fig.add_annotation(text="Missing Library: pip install umap-learn", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="red"))
            return fig
        if len(y_names) < 3:
            fig.add_annotation(text="Please select at least 3 series for UMAP clustering.", xref="paper",
                               yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=18))
            return fig

        z_safe = np.nan_to_num(z_data, nan=0.0)
        n_neighbors = min(umap_neigh, max(2, len(y_names) - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=umap_dist, random_state=42)
        embedding = reducer.fit_transform(z_safe)

        unique_cats = sorted(list(set([lob_to_category.get(name, "Unknown") for name in y_names])))
        for cat in unique_cats:
            cat_indices = [idx for idx, name in enumerate(y_names) if lob_to_category.get(name, "Unknown") == cat]
            fig.add_trace(go.Scatter(
                x=embedding[cat_indices, 0], y=embedding[cat_indices, 1],
                mode="markers+text", name=cat,
                text=[y_names[idx] for idx in cat_indices],
                textposition="top center", textfont=dict(size=10, color="rgba(0,0,0,0.6)"),
                marker=dict(size=12, line=dict(width=1, color="White")),
                customdata=[y_names[idx] for idx in cat_indices],
                hovertemplate="<b>%{customdata}</b><br>Category: " + cat + "<br>UMAP-1: %{x:.2f}<br>UMAP-2: %{y:.2f}<extra></extra>"
            ))
        fig.update_layout(
            title=f"UMAP Projection (neighbors={n_neighbors}, min_dist={umap_dist:.2f})", height=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            plot_bgcolor="#f9f9f9", hovermode="closest",
            legend=dict(title="Categories", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                        itemclick=False, itemdoubleclick=False)
        )

    elif graph_type == "t-SNE Clusters":
        if not SKLEARN_AVAILABLE:
            fig.add_annotation(text="Missing Library: pip install scikit-learn", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="red"))
            return fig
        if len(y_names) < 3:
            fig.add_annotation(text="Please select at least 3 series for t-SNE clustering.", xref="paper",
                               yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=18))
            return fig

        z_safe = np.nan_to_num(z_data, nan=0.0)
        perplexity_val = min(tsne_perp, max(1, len(y_names) - 1))
        tsne_model = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init="pca",
                          learning_rate="auto")
        embedding = tsne_model.fit_transform(z_safe)

        unique_cats = sorted(list(set([lob_to_category.get(name, "Unknown") for name in y_names])))
        for cat in unique_cats:
            cat_indices = [idx for idx, name in enumerate(y_names) if lob_to_category.get(name, "Unknown") == cat]
            fig.add_trace(go.Scatter(
                x=embedding[cat_indices, 0], y=embedding[cat_indices, 1],
                mode="markers+text", name=cat,
                text=[y_names[idx] for idx in cat_indices],
                textposition="top center", textfont=dict(size=10, color="rgba(0,0,0,0.6)"),
                marker=dict(size=12, line=dict(width=1, color="White")),
                customdata=[y_names[idx] for idx in cat_indices],
                hovertemplate="<b>%{customdata}</b><br>Category: " + cat + "<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<extra></extra>"
            ))
        fig.update_layout(
            title=f"t-SNE Projection (perplexity={perplexity_val})", height=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            plot_bgcolor="#f9f9f9", hovermode="closest",
            legend=dict(title="Categories", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                        itemclick=False, itemdoubleclick=False)
        )

    elif graph_type == "3D Lines":
        vibrant_palette = (
                                  px.colors.qualitative.Plotly + px.colors.qualitative.Vivid[:-1] +
                                  px.colors.qualitative.Prism[:-1] + px.colors.qualitative.Set1[:-2] +
                                  px.colors.qualitative.Set2 + px.colors.qualitative.Set3
                          ) * 10
        for i, name in enumerate(y_names):
            fig.add_trace(go.Scatter3d(
                x=x_data, y=[name] * len(x_data), z=z_data[i], name=name, mode="lines",
                line=dict(width=4, color=vibrant_palette[i]), customdata=[name] * len(x_data)
            ))
        fig.update_layout(
            title=title_text, height=fig_height,
            scene=dict(
                xaxis=dict(title="Time", range=[len(x_data) - 0.5, -0.5], autorange="reversed"),
                yaxis=dict(title="Day/Product"),
                zaxis=dict(title=value_title)
            ),
            clickmode="event+select", plot_bgcolor="#f9f9f9", margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True,
            legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02,
                    "itemclick": False, "itemdoubleclick": False}
        )

    elif graph_type == "Horizon Chart":
        num_bands = 5
        pos_colors = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
        neg_colors = ["#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]

        valid_z = np.array(z_data, dtype=float).flatten()
        valid_z = valid_z[~np.isnan(valid_z)]
        actual_max_abs = np.nanmax(np.abs(valid_z)) if len(valid_z) else 1
        if actual_max_abs == 0 or np.isnan(actual_max_abs):
            actual_max_abs = 1

        _, padded_max_abs, pos_ticks, pos_labels = calculate_nice_ticks(0, actual_max_abs, 4)
        band_size = padded_max_abs / num_bands

        for i, name in enumerate(y_names):
            z = np.array(z_data[i], dtype=float)
            z_safe = np.nan_to_num(z, nan=0.0)
            custom_data_arr = [name] * len(x_data)

            fig.add_trace(go.Scatter(
                x=x_data, y=[i + 0.5] * len(x_data), mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=1), name=name, customdata=custom_data_arr,
                cliponaxis=False,
                text=[f"{val:.4f}" if not np.isnan(val) else "NaN" for val in z],
                hovertemplate="<b>%{customdata}</b><br>Time: %{x}<br>Value: %{text}<extra></extra>",
                hoverinfo="all", showlegend=False
            ))

            for b in range(num_bands):
                pos_vals = np.clip(z_safe - b * band_size, 0, band_size)
                fig.add_trace(go.Scatter(x=x_data, y=[i] * len(x_data), mode="lines",
                                         line=dict(width=0), showlegend=False, hoverinfo="skip",
                                         customdata=custom_data_arr, cliponaxis=False))
                fig.add_trace(go.Scatter(x=x_data, y=i + (pos_vals / band_size), mode="lines",
                                         fill="tonexty", fillcolor=pos_colors[b], line=dict(width=0),
                                         showlegend=False, hoverinfo="skip", customdata=custom_data_arr,
                                         cliponaxis=False))

                neg_vals = np.clip(-z_safe - b * band_size, 0, band_size)
                fig.add_trace(go.Scatter(x=x_data, y=[i] * len(x_data), mode="lines",
                                         line=dict(width=0), showlegend=False, hoverinfo="skip",
                                         customdata=custom_data_arr, cliponaxis=False))
                fig.add_trace(go.Scatter(x=x_data, y=i + (neg_vals / band_size), mode="lines",
                                         fill="tonexty", fillcolor=neg_colors[b], line=dict(width=0),
                                         showlegend=False, hoverinfo="skip", customdata=custom_data_arr,
                                         cliponaxis=False))

            fig.add_trace(go.Scatter(x=x_data, y=[i] * len(x_data), mode="lines",
                                     line=dict(width=1, color="#444444"), showlegend=False, hoverinfo="skip",
                                     customdata=custom_data_arr, cliponaxis=False))

        custom_colorscale = [
            [0.0, neg_colors[4]], [0.1, neg_colors[4]], [0.1, neg_colors[3]], [0.2, neg_colors[3]],
            [0.2, neg_colors[2]], [0.3, neg_colors[2]], [0.3, neg_colors[1]], [0.4, neg_colors[1]],
            [0.4, neg_colors[0]], [0.5, neg_colors[0]], [0.5, pos_colors[0]], [0.6, pos_colors[0]],
            [0.6, pos_colors[1]], [0.7, pos_colors[1]], [0.7, pos_colors[2]], [0.8, pos_colors[2]],
            [0.8, pos_colors[3]], [0.9, pos_colors[3]], [0.9, pos_colors[4]], [1.0, pos_colors[4]]
        ]
        tickvals = [-t for t in reversed(pos_ticks[1:])] + pos_ticks
        ticktext = [f"-{l}" if l != "0" else "0" for l in reversed(pos_labels[1:])] + pos_labels

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(colorscale=custom_colorscale, cmin=-padded_max_abs, cmax=padded_max_abs, showscale=True,
                        colorbar=dict(title=value_title, thickness=15, outlinewidth=0, lenmode="pixels", len=250,
                                      yanchor="top", y=1, tickmode="array", tickvals=tickvals, ticktext=ticktext)),
            showlegend=False, hoverinfo="none"
        ))
        fig.update_layout(
            title=title_text, height=fig_height,
            xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
            yaxis={"title": "Day/Product", "tickvals": [i + 0.5 for i in range(len(y_names))],
                   "ticktext": y_names, "range": [-0.5, max(1, len(y_names))], "automargin": True},
            clickmode="event+select", hovermode="closest", plot_bgcolor="#f9f9f9", showlegend=False,
            margin=dict(t=70, b=40, l=80, r=20)
        )

    elif graph_type == "Spark Line":
        valid_z = np.array(z_data, dtype=float).flatten()
        valid_z = valid_z[~np.isnan(valid_z)]
        max_abs = np.nanmax(np.abs(valid_z)) if len(valid_z) else 1
        if max_abs == 0 or np.isnan(max_abs):
            max_abs = 1

        trace_index = 0
        data_line_indices = []

        for i, name in enumerate(y_names):
            z = np.array(z_data[i], dtype=float)
            custom_data_arr = [name] * len(x_data)

            fig.add_trace(go.Scatter(
                x=x_data, y=[i] * len(x_data), mode="lines",
                line=dict(width=1, color="#444444"), showlegend=False, hoverinfo="skip",
                customdata=custom_data_arr, cliponaxis=False
            ))
            trace_index += 1

            baseline_gaps = [i if not np.isnan(val) else None for val in z]
            valid_mask = ~np.isnan(z)

            if np.any(valid_mask):
                if np.max(z[valid_mask]) > 0:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=baseline_gaps, mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
                        showlegend=False, hoverinfo="skip", customdata=custom_data_arr,
                        connectgaps=False, cliponaxis=False
                    ))
                    trace_index += 1
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=i + (np.where(np.isnan(z), np.nan, np.clip(z, 0, None)) / max_abs) * 0.45,
                        mode="lines", fill="tonexty", fillcolor="rgba(31, 119, 180, 0.4)",
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                        customdata=custom_data_arr, connectgaps=False, cliponaxis=False
                    ))
                    trace_index += 1

                if np.min(z[valid_mask]) < 0:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=baseline_gaps, mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
                        showlegend=False, hoverinfo="skip", customdata=custom_data_arr,
                        connectgaps=False, cliponaxis=False
                    ))
                    trace_index += 1
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=i + (np.where(np.isnan(z), np.nan, np.clip(z, None, 0)) / max_abs) * 0.45,
                        mode="lines", fill="tonexty", fillcolor="rgba(214, 39, 40, 0.4)",
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                        customdata=custom_data_arr, connectgaps=False, cliponaxis=False
                    ))
                    trace_index += 1

                real_z = np.where(np.isnan(z), np.nan, z)
                text_vals = [f"{val:.4f}" if not np.isnan(val) else "NaN" for val in real_z]

                data_line_indices.append(trace_index)
                fig.add_trace(go.Scatter(
                    x=x_data, y=i + (real_z / max_abs) * 0.45, mode="lines+markers",
                    line=dict(width=1.5, color="#bbbbbb"),
                    marker=dict(size=5, color=real_z,
                                colorscale=[[0, "#d62728"], [0.45, "#bbbbbb"], [0.55, "#bbbbbb"], [1, "#1f77b4"]],
                                cmin=-max_abs, cmax=max_abs, showscale=False, line=dict(width=0)),
                    name=name, text=text_vals, customdata=custom_data_arr,
                    hovertemplate="<b>%{customdata}</b><br>Time: %{x}<br>Value: %{text}<extra></extra>",
                    showlegend=False, hoverinfo="all", connectgaps=False, cliponaxis=False
                ))
                trace_index += 1

        fig.update_layout(
            title=title_text, height=fig_height,
            xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
            yaxis={"title": "Day/Product", "tickvals": [i + 0.5 for i in range(len(y_names))],
                   "ticktext": y_names, "range": [-0.5, max(1, len(y_names))], "automargin": True},
            clickmode="event+select", hovermode="closest", plot_bgcolor="#f9f9f9", showlegend=False,
            margin=dict(t=70, b=40, l=80, r=20),
            updatemenus=[dict(
                type="buttons", direction="left",
                buttons=[
                    dict(args=[{"mode": "lines"}, data_line_indices], label="Hide Markers", method="restyle"),
                    dict(args=[{"mode": "lines+markers"}, data_line_indices], label="Show Markers", method="restyle")
                ],
                pad={"r": 10, "t": 10}, showactive=True, x=1.0, xanchor="right", y=1.05, yanchor="bottom",
                bgcolor="#ffffff", bordercolor="#007bff", font=dict(size=12, color="#007bff")
            )]
        )

    else:  # Line Chart
        vibrant_palette = (
                                  px.colors.qualitative.Plotly + px.colors.qualitative.Vivid[:-1] +
                                  px.colors.qualitative.Prism[:-1] + px.colors.qualitative.Set1[:-2] +
                                  px.colors.qualitative.Set2 + px.colors.qualitative.Set3
                          ) * 10
        for i, name in enumerate(y_names):
            fig.add_trace(go.Scatter(
                x=x_data, y=z_data[i], name=name, mode="lines",
                line=dict(color=vibrant_palette[i]), customdata=[name] * len(x_data)
            ))
        fig.update_layout(
            title=title_text, height=fig_height,
            xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
            yaxis={"title": value_title},
            showlegend=True,
            legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02,
                    "itemclick": False, "itemdoubleclick": False},
            clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9"
        )

    return fig


def main():
    global lob_to_category, sorted_categories
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    print("Loading data...")
    level_depth = 5
    all_data, names, detections = load_all_data(level_depth=level_depth)

    if not all_data:
        print("Warning: No data loaded. Make sure 'data/' has CSV files.")

    # Load optional category mapping
    lob_to_category = fetch_categories()
    for name in names:
        if name not in lob_to_category:
            lob_to_category[name] = "Unknown"
    sorted_categories = sorted(list(set(lob_to_category.values())))

    placeholder_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))
    placeholder_fig.update_layout(annotations=[{
        "text": "Select Day/Product using the dropdown above to view the price graph",
        "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 20}
    }])
    placeholder_fig.register_update_graph_callback(app=app, graph_id="price_graph")
    bottom_fig = go.Figure()

    initial_lob_options = [{"label": name, "value": name} for name in names]

    header_block = html.Div([
        html.H2("LOB Visualization Dashboard",
                style={"margin": "0", "color": "#ffffff", "fontWeight": "bold"}),
        html.P("Order Book Analysis | Anomaly Detection",
               style={"margin": "0", "marginTop": "5px", "color": "#d0d0d0", "fontSize": "0.95rem"})
    ], style={"backgroundColor": "#343a40", "padding": "1.2rem", "borderRadius": "10px",
              "marginBottom": "1rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"})

    app.layout = html.Div([
        header_block,

        # --- TOP GRAPH ---
        html.Div([
            html.Div([
                html.P("Select LOB (Top Graph):", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="lob_selector", options=initial_lob_options,
                    value=names[0] if names else None, clearable=False,
                    persistence=True, persistence_type="local"
                ),
            ], style={"marginBottom": "1rem", "padding": "0.5rem", "backgroundColor": "#ffffff",
                      "borderRadius": "10px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),
            dcc.Loading(
                dcc.Graph(id="price_graph", figure=placeholder_fig,
                          config={"toImageButtonOptions": {"format": "png", "filename": "price_graph",
                                                           "width": 1920, "height": 1080, "scale": 3}}),
                type="circle"
            )
        ], style={"marginBottom": "1rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                  "borderRadius": "10px", "padding": "0.5rem", "backgroundColor": "#f9f9f9"}),

        # --- BOTTOM SECTION: sidebar + graph ---
        html.Div([
            # Sidebar
            html.Div([
                html.P("Select a metric:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="metric_dropdown",
                    options=[{"label": label, "value": value} for label, value in [
                        ("Ask Price", "Ask Price 1"), ("Bid Price", "Bid Price 1"),
                        ("Ask Volume", "Ask Volume 1"), ("Bid Volume", "Bid Volume 1"),
                        ("Imbalance Index", "Imbalance Index"),
                        ("Frequency of Incoming Messages", "Frequency of Incoming Messages"),
                        ("Cancellations Rate", "Cancellations Rate"),
                        ("High Quoting Activity", "High Quoting Activity"),
                        ("Unbalanced Quoting", "Unbalanced Quoting"),
                        ("Low Execution Probability", "Low Execution Probability"),
                        ("Trades Oppose Quotes", "Trades Oppose Quotes"),
                        ("Cancels Oppose Trades", "Cancels Oppose Trades")
                    ]],
                    value=metric, clearable=False, persistence=True, persistence_type="local",
                    style={"marginBottom": "0.5rem"}
                ),
                html.P(id="metric_description", style={"marginBottom": "1rem", "fontStyle": "italic"}),

                html.P("Select an aggregation function:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="aggregation_dropdown",
                    options=[{"label": x, "value": x} for x in ["Mean", "Median", "Max", "Min", "Std"]],
                    value=chosen_aggregation, clearable=False, persistence=True, persistence_type="local",
                    style={"marginBottom": "1rem"}
                ),

                html.P("Select a time window (minutes):", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Input(
                    id="time_window_input", type="number", value=time_window_aggregation,
                    min=1, max=720, step=1, placeholder="Time Window (minutes)",
                    persistence=True, persistence_type="local",
                    style={"width": "100%", "marginBottom": "1rem"}
                ),

                html.P("Data Processing:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="data_processing_dropdown",
                    options=[
                        {"label": "None (Raw Values)", "value": "none"},
                        {"label": "Normalize (Min-Max)", "value": "normalize"},
                        {"label": "Percentage Change (%)", "value": "pct_change"}
                    ],
                    value="normalize", clearable=False, persistence=True, persistence_type="local",
                    style={"marginBottom": "1rem"}
                ),

                html.P("Select bottom graph style:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="graph_type_dropdown",
                    options=[
                        {"label": " Heatmap", "value": "Heatmap"},
                        {"label": " 2D Line chart", "value": "Line Chart"},
                        {"label": " 3D Line chart", "value": "3D Lines"},
                        {"label": " Horizon Chart", "value": "Horizon Chart"},
                        {"label": " Spark Line", "value": "Spark Line"},
                        {"label": " Correlation Matrix", "value": "Correlation Matrix"},
                        {"label": " UMAP Clusters", "value": "UMAP Clusters"},
                        {"label": " t-SNE Clusters", "value": "t-SNE Clusters"}
                    ],
                    value="Heatmap", clearable=False, persistence=True, persistence_type="local",
                    style={"marginBottom": "1rem"}
                ),

                html.P("Sort Series By:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="sort_by_dropdown",
                    options=[
                        {"label": "Default (Original Order)", "value": "Default"},
                        {"label": "Alphabetical (A-Z)", "value": "Alphabetical (A-Z)"},
                        {"label": "Alphabetical (Z-A)", "value": "Alphabetical (Z-A)"},
                        {"label": "Correlation (Group Similar)", "value": "Correlation"}
                    ],
                    value="Default", clearable=False, persistence=True, persistence_type="local",
                    style={"marginBottom": "1rem"}
                ),

                # --- CORRELATION FILTER PANEL ---
                html.P("Correlation Filter Target:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="corr_reference_lob",
                    options=[{"label": "None (Disabled)", "value": "NONE"}] + initial_lob_options,
                    value="NONE", clearable=False, persistence=True, persistence_type="local",
                    style={"marginBottom": "0.5rem"}
                ),
                html.Div([
                    html.P("Correlation Range to INCLUDE:",
                           style={"fontWeight": "bold", "marginTop": "0.5rem", "marginBottom": "0.2rem",
                                  "fontSize": "0.85rem", "color": "#6c757d"}),
                    html.Div([
                        dcc.RangeSlider(
                            id="corr_range_slider", min=-1.0, max=1.0, step=0.05, value=[0.5, 1.0],
                            marks={-1: "-1", -0.5: "-0.5", 0: "0", 0.5: "0.5", 1: "1"},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={"padding": "0 10px", "marginBottom": "0.8rem"}),
                    html.Div(id="corr_live_preview",
                             style={"fontSize": "0.85rem", "fontStyle": "italic", "color": "#17a2b8",
                                    "textAlign": "center", "marginBottom": "1rem"})
                ], id="corr_filter_controls_container", style={"display": "none"}),
                # ----------------------------------------

                html.P("Filter by Category:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="category_selector",
                    options=[{"label": "All Categories", "value": "ALL"}] + [
                        {"label": s, "value": s} for s in sorted_categories
                    ],
                    value="ALL", clearable=False, persistence=True, persistence_type="local",
                    style={"marginBottom": "1.5rem"}
                ),

                html.P("Filter Specific Series:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="timeseries_selector", options=initial_lob_options, value=names.copy(),
                    multi=True, placeholder="Select series...",
                    persistence=True, persistence_type="local",
                    style={"marginBottom": "0.5rem"}
                ),
                html.Div([
                    html.Button("Select All", id="select_all_btn", n_clicks=0, style={
                        "width": "48%", "marginRight": "4%", "padding": "0.4rem",
                        "backgroundColor": "#007bff", "color": "white", "border": "none", "borderRadius": "6px"
                    }),
                    html.Button("Unselect All", id="unselect_all_btn", n_clicks=0, style={
                        "width": "48%", "padding": "0.4rem",
                        "backgroundColor": "#dc3545", "color": "white", "border": "none", "borderRadius": "6px"
                    }),
                ], style={"display": "flex", "marginBottom": "1rem"}),

                html.Button("Apply", id="update_bottom_graph_button", n_clicks=0, style={
                    "marginTop": "1rem", "width": "100%", "padding": "0.75rem 1rem",
                    "fontSize": "1rem", "fontWeight": "bold", "color": "#fff",
                    "backgroundColor": "#28a745", "border": "none", "borderRadius": "8px",
                    "boxShadow": "0 4px 6px rgba(40, 167, 69, 0.3)", "cursor": "pointer"
                }),
            ], style={
                "flex": "0 0 300px", "padding": "0.5rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                "borderRadius": "10px", "backgroundColor": "#ffffff", "minWidth": "250px",
                "position": "sticky", "top": "1rem"
            }),

            # Main graph area
            html.Div([
                dcc.Loading(html.Div([
                    # t-SNE overlay params
                    html.Div([
                        html.P("t-SNE Perplexity:",
                               style={"fontWeight": "bold", "marginBottom": "0.2rem", "fontSize": "0.9rem"}),
                        dcc.Slider(id="tsne_perplexity", min=2, max=50, step=1, value=30,
                                   marks={2: "2", 50: "50"},
                                   tooltip={"placement": "bottom", "always_visible": True},
                                   updatemode="mouseup"),
                    ], id="tsne_params", style={
                        "display": "none", "position": "absolute", "top": "70px", "left": "15px",
                        "width": "280px", "zIndex": "1000", "background": "rgba(255,255,255,0.9)",
                        "padding": "10px", "borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0,0,0,0.2)",
                        "border": "1px solid #dee2e6"
                    }),

                    # UMAP overlay params
                    html.Div([
                        html.P("UMAP Neighbors:",
                               style={"fontWeight": "bold", "marginBottom": "0.2rem", "fontSize": "0.9rem"}),
                        dcc.Slider(id="umap_neighbors", min=2, max=50, step=1, value=15,
                                   marks={2: "2", 50: "50"},
                                   tooltip={"placement": "bottom", "always_visible": True},
                                   updatemode="mouseup"),
                        html.P("UMAP Min Distance:",
                               style={"fontWeight": "bold", "marginBottom": "0.2rem", "marginTop": "0.8rem",
                                      "fontSize": "0.9rem"}),
                        dcc.Slider(id="umap_min_dist", min=0.0, max=0.99, step=0.05, value=0.1,
                                   marks={0: "0", 1: "1"},
                                   tooltip={"placement": "bottom", "always_visible": True},
                                   updatemode="mouseup"),
                    ], id="umap_params", style={
                        "display": "none", "position": "absolute", "top": "70px", "left": "15px",
                        "width": "280px", "zIndex": "1000", "background": "rgba(255,255,255,0.9)",
                        "padding": "10px", "borderRadius": "8px", "boxShadow": "0 2px 5px rgba(0,0,0,0.2)",
                        "border": "1px solid #dee2e6"
                    }),

                    dcc.Graph(
                        id="bottom_graph", figure=bottom_fig, clear_on_unhover=True,
                        config={"toImageButtonOptions": {"format": "png", "filename": "bottom_graph",
                                                         "width": 1920, "height": 1080, "scale": 3}}
                    )
                ], style={"position": "relative", "width": "100%", "height": "100%"}), type="circle")
            ], style={
                "flex": "1", "marginLeft": "1rem", "padding": "0.5rem",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "borderRadius": "10px",
                "backgroundColor": "#f9f9f9", "minWidth": "0", "maxHeight": "100vh", "overflowY": "auto"
            })
        ], style={"display": "flex", "flexWrap": "nowrap", "alignItems": "flex-start", "gap": "1rem"}),

        html.Div(id="dummy-clientside-output", style={"display": "none"})
    ], style={"padding": "0.5rem", "fontFamily": "Arial, sans-serif", "backgroundColor": "#f0f2f5"})

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    # Toggle UMAP / t-SNE param panels
    @app.callback(
        [Output("tsne_params", "style"), Output("umap_params", "style")],
        Input("graph_type_dropdown", "value"),
        State("tsne_params", "style"), State("umap_params", "style")
    )
    def toggle_cluster_params(gtype, current_tsne_style, current_umap_style):
        tsne_style = (current_tsne_style or {}).copy()
        tsne_style["display"] = "block" if gtype == "t-SNE Clusters" else "none"
        umap_style = (current_umap_style or {}).copy()
        umap_style["display"] = "block" if gtype == "UMAP Clusters" else "none"
        return tsne_style, umap_style

    # Correlation filter panel visibility + live preview
    @app.callback(
        [Output("corr_filter_controls_container", "style"),
         Output("corr_live_preview", "children")],
        [Input("corr_reference_lob", "value"),
         Input("corr_range_slider", "value"),
         Input("timeseries_selector", "value")],
        [State("metric_dropdown", "value"),
         State("aggregation_dropdown", "value"),
         State("time_window_input", "value"),
         State("data_processing_dropdown", "value")]
    )
    def update_corr_ui_and_preview(
            corr_ref,
            inclusion_range,
            selected_timeseries,
            selected_metric,
            selected_aggregation,
            selected_time_window,
            data_processing
    ):
        if not corr_ref or corr_ref == "NONE":
            return {"display": "none"}, ""

        if not selected_timeseries:
            return {"display": "block"}, "No items selected."

        selected_time_window_sec = (selected_time_window or 60) * MINUTE_SEC

        new_z = aggregate_data(
            all_data,
            metric=selected_metric,
            aggregation=aggregation_functions_map[selected_aggregation],
            time_window=selected_time_window_sec
        )

        if data_processing == "normalize":
            new_z = np.array(normalize_data(new_z))
        elif data_processing == "pct_change":
            new_z = np.array(pct_change_data(new_z))

        selected_indices = [
            i for i, n in enumerate(names)
            if n in selected_timeseries
        ]

        filtered_names = [names[i] for i in selected_indices]
        filtered_z = np.array(new_z)[selected_indices]

        kept_names, _, valid_count = filter_by_correlation(
            filtered_names,
            filtered_z,
            corr_ref,
            inclusion_range
        )

        return {
            "display": "block"
        }, f"Preview: Keeping {len(kept_names)} of {valid_count} valid items."

    # Category filter + Select All / Unselect All
    @app.callback(
        [Output("timeseries_selector", "options"), Output("timeseries_selector", "value")],
        Input("category_selector", "value"),
        Input("select_all_btn", "n_clicks"),
        Input("unselect_all_btn", "n_clicks"),
        State("timeseries_selector", "value")
    )
    def update_series_dropdown(selected_category, select_all, unselect_all, current_selection):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        available = names if selected_category == "ALL" else [
            n for n in names if lob_to_category.get(n) == selected_category
        ]
        options = [{"label": n, "value": n} for n in available]

        if trigger == "unselect_all_btn":
            return options, []
        elif trigger in ("select_all_btn", "category_selector"):
            return options, available
        else:
            valid = [s for s in (current_selection or []) if s in available]
            return options, valid

    # Metric description
    @app.callback(Output("metric_description", "children"), Input("metric_dropdown", "value"))
    def update_description(selected_metric):
        return metric_descriptions_map.get(selected_metric, "No description available.")

    # Top price graph
    @app.callback(Output("price_graph", "figure"), Input("lob_selector", "value"))
    def update_price_graph_from_dropdown(selected_name):
        if not selected_name:
            return dash.no_update

        idx = names.index(selected_name)
        data = all_data[idx]

        timestamps = data["Time"].values
        ask_prices = [data[f"Ask Price {i}"].values for i in range(1, level_depth + 1)
                      if f"Ask Price {i}" in data.columns]
        bid_prices = [data[f"Bid Price {i}"].values for i in range(1, level_depth + 1)
                      if f"Bid Price {i}" in data.columns]
        imbalance_indices = data["Imbalance Index"].values
        freqs = data["Frequency of Incoming Messages"].values
        cancels = data["Cancellations Rate"].values

        fig = create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels,
                                 selected_name, detections[idx])
        fig.register_update_graph_callback(app=app, graph_id="price_graph")
        return fig

    # Bottom graph
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
        Input("corr_reference_lob", "value"),
        Input("corr_range_slider", "value"),
        Input("metric_dropdown", "value"),
        Input("aggregation_dropdown", "value"),
        Input("time_window_input", "value"),
        State("bottom_graph", "figure")
    )
    def update_bottom_graph(btn_clicks, selected_graph_type, data_processing, selected_timeseries, sort_by,
                            tsne_perp, umap_neigh, umap_dist,
                            corr_ref_stock, corr_range,
                            selected_metric, selected_aggregation, selected_time_window, current_bottom_fig):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        selected_time_window_sec = (selected_time_window or time_window_aggregation) * MINUTE_SEC
        new_x = [
            f"{int(i // HOUR_SEC):02d}:{int(i % HOUR_SEC // MINUTE_SEC):02d}"
            for i in range(0, DAY_SEC, selected_time_window_sec)
        ]
        new_z = aggregate_data(all_data, metric=selected_metric,
                               aggregation=aggregation_functions_map[selected_aggregation],
                               time_window=selected_time_window_sec)

        if data_processing == "normalize":
            new_z = normalize_data(new_z)
        elif data_processing == "pct_change":
            new_z = pct_change_data(new_z)

        # Filter by selected series
        if selected_timeseries:
            selected_indices = [i for i, n in enumerate(names) if n in selected_timeseries]
            filtered_names = [names[i] for i in selected_indices]
            filtered_z = np.array(new_z)[selected_indices]

            # --- CORRELATION FILTERING ---
            filtered_names, filtered_z, _ = filter_by_correlation(
                filtered_names,
                filtered_z,
                corr_ref_stock,
                corr_range
            )

            # Sorting
            if sort_by == "Alphabetical (A-Z)" and len(filtered_z) > 0:
                sort_idx = np.argsort(filtered_names)
                filtered_names = [filtered_names[i] for i in sort_idx]
                filtered_z = filtered_z[sort_idx]
            elif sort_by == "Alphabetical (Z-A)" and len(filtered_z) > 0:
                sort_idx = np.argsort(filtered_names)[::-1]
                filtered_names = [filtered_names[i] for i in sort_idx]
                filtered_z = filtered_z[sort_idx]
            elif sort_by == "Correlation" and len(filtered_z) > 1:
                n = len(filtered_names)
                corr_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        val = safe_corr(filtered_z[i], filtered_z[j])
                        corr_matrix[i, j] = val
                        corr_matrix[j, i] = val

                # Fill diagonal with low value so max() avoids self
                np.fill_diagonal(corr_matrix, -2)

                unvisited = set(range(len(filtered_names)))
                sort_idx = [0]
                unvisited.remove(0)
                curr_idx = 0
                while unvisited:
                    next_idx = max(unvisited, key=lambda i: corr_matrix[curr_idx, i])
                    sort_idx.append(next_idx)
                    unvisited.remove(next_idx)
                    curr_idx = next_idx
                filtered_names = [filtered_names[i] for i in sort_idx]
                filtered_z = filtered_z[sort_idx]
        else:
            filtered_names = []
            filtered_z = np.array([])

        # Preserve x-axis zoom across non-apply triggers
        current_range = None
        if current_bottom_fig and isinstance(current_bottom_fig, dict):
            layout = current_bottom_fig.get("layout", {})
            current_range = layout.get("xaxis", {}).get("range")

        new_bottom_fig = create_bottom_figure(
            filtered_z, new_x, filtered_names, selected_graph_type,
            selected_metric, selected_aggregation, data_processing,
            tsne_perp, umap_neigh, umap_dist
        )

        if trigger_id not in ("update_bottom_graph_button", "graph_type_dropdown") and current_range:
            if selected_graph_type == "3D Lines":
                new_bottom_fig.update_layout(scene=dict(xaxis={"range": current_range}))
            elif selected_graph_type not in ("Correlation Matrix", "UMAP Clusters", "t-SNE Clusters"):
                new_bottom_fig.update_layout(xaxis={"range": current_range})

        return new_bottom_fig

    # Multi-select legend click + hover highlighting (clientside)
    app.clientside_callback(
        """
        function(figure) {
            setTimeout(function() {
                const graph = document.getElementById('bottom_graph');
                if (!graph) return;
                const plot = graph.querySelector('.js-plotly-plot');
                if (!plot) return;

                plot._selectedTraces = new Set();
                plot._originalColors = null;

                function updateStyles(hoveredIndex = -1) {
                    const traceCount = plot.data.length;
                    const isCluster = traceCount > 0 && (
                        plot.data[0].mode === 'markers+text' || plot.data[0].mode === 'markers'
                    );

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
                            update.opacity.push(1);
                            if (!isCluster) update.line.push({ width: is3D ? 4 : 2, color: plot._originalColors[i] });
                        } else if (isActive) {
                            update.opacity.push(1);
                            if (!isCluster) update.line.push({ width: is3D ? 8 : 5, color: plot._originalColors[i] });
                        } else {
                            update.opacity.push(is3D ? 0.4 : 0.25);
                            if (!isCluster) update.line.push({ width: is3D ? 2.5 : 1.5, color: "rgba(130,130,130,0.9)" });
                        }
                    }

                    Plotly.restyle(plot, update).then(() => {
                        if (isCluster || traceCount === 0 || plot.data[0].type === 'scatter3d') return;
                        const scatterLayer = plot.querySelector('.scatterlayer');
                        if (scatterLayer) {
                            const traces = Array.from(scatterLayer.querySelectorAll('.trace.scatter'));
                            const uidToIndex = {};
                            (plot._fullData || plot.data).forEach((t, idx) => {
                                if (t.uid) uidToIndex[t.uid] = idx;
                            });
                            traces.forEach(node => {
                                const className = node.getAttribute('class') || '';
                                let originalIdx = -1;
                                for (let uid in uidToIndex) {
                                    if (className.includes(uid)) { originalIdx = uidToIndex[uid]; break; }
                                }
                                if (originalIdx !== -1 &&
                                    (plot._selectedTraces.has(originalIdx) || originalIdx === hoveredIndex)) {
                                    scatterLayer.appendChild(node);
                                }
                            });
                        }
                    });
                }

                function attachLegendEvents() {
                    const legendItems = plot.querySelectorAll('.legend .traces');
                    if (!legendItems.length) return;
                    legendItems.forEach((item, index) => {
                        item.onmouseenter = null; item.onmouseleave = null; item.onclick = null;
                        item.onmouseenter = function() { updateStyles(index); };
                        item.onmouseleave = function() { updateStyles(-1); };
                        item.onclick = function(e) {
                            e.stopPropagation(); e.preventDefault();
                            if (plot._selectedTraces.has(index)) plot._selectedTraces.delete(index);
                            else plot._selectedTraces.add(index);
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
    level_depth = 5
    print("Starting...")
    main()