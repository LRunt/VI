import os
import sys
import json
import math
import numpy as np
import pandas as pd
import datetime

import plotly.graph_objects as go
import plotly_resampler
from plotly_resampler import FigureResampler

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# --- Data Loading Logic ---

# Minute in seconds (60 seconds)
MINUTE_SEC = 60
# Hour in seconds (60 minutes)
HOUR_SEC = 60 * MINUTE_SEC
# Day in seconds (24 hours)
DAY_SEC = 24 * HOUR_SEC


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
    if not os.path.exists(os.path.join(res_dir)):
        return None
    files = os.listdir(res_dir)
    if not files or files == []:
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

        if lower_is_better:
            anomaly_proba = y_scores_norm
        else:
            anomaly_proba = 1 - y_scores_norm

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


def aggregate_data(all_data, metric="Ask Price 1", aggregation=np.mean, time_window=3600):
    aggregated_data = []

    for data in all_data:
        tmp_agg = []
        timestamps = pd.to_datetime(data["Time"].values, unit="ns")
        timestamps_series = pd.Series(timestamps)
        seconds_since_midnight = (timestamps_series - timestamps_series.dt.normalize()).dt.total_seconds()

        for i in range(0, DAY_SEC, time_window):
            start_time = i
            end_time = i + time_window

            data_in_window = data[(seconds_since_midnight >= start_time) & (seconds_since_midnight < end_time)]
            data_in_window = data_in_window[metric].dropna()

            if data_in_window.empty:
                tmp_agg.append(np.nan)
                continue
            tmp_agg.append(aggregation(data_in_window.values))

        tmp_agg = np.array(tmp_agg)
        aggregated_data.append(tmp_agg)

    aggregated_data = np.array(aggregated_data)
    return aggregated_data


# --- Dashboard Logic ---

HOST_ADDRESS = "127.0.0.1"
PORT = 8080

timestamps_graph_labels = None
last_update_bottom_graph_click_count = None
last_graph_type = "Heatmap"
last_data_processing_state = "normalize"

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


def normalize_data(z_data):
    normalized_z = []
    for row in z_data:
        arr = np.array(row, dtype=np.float64)
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if max_val > min_val:
            norm_arr = (arr - min_val) / (max_val - min_val)
        else:
            norm_arr = np.zeros_like(arr)
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


def create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels, name, detected_anomalies,
                       how_many_x_ticks=75):
    global timestamps_graph_labels
    timestamps_graph_labels = [datetime.datetime.fromtimestamp(int(ts) / 1e9 - HOUR_SEC).strftime("%H:%M:%S.%f") for ts
                               in timestamps]
    timestamps_graph = list(range(len(timestamps_graph_labels)))
    tickvals = list(range(0, len(timestamps), max(1, len(timestamps) // how_many_x_ticks)))
    ticklabels = [timestamps_graph_labels[i] for i in tickvals]
    ticklabels = [ts[:8] for ts in ticklabels]

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

        x_vals_with_Nones = []
        y_vals_with_Nones = []
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


def create_bottom_figure(z_data, x_data, y_names, graph_type, metric_name, agg_name, data_processing="none"):
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

    if graph_type == "Heatmap":
        fig.add_trace(
            go.Heatmap(
                z=z_data, x=x_data, y=y_names, colorscale="Viridis",
                colorbar=dict(), hoverongaps=False, zmin=np.nanmin(z_data) if len(z_data) else 0,
                zmax=np.nanmax(z_data) if len(z_data) else 1,
            )
        )
        fig.update_layout(
            title=title_text, xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
            yaxis={"title": "Day/Product"}, clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9"
        )

    elif graph_type == "3D Lines":
        for i, name in enumerate(y_names):
            fig.add_trace(
                go.Scatter3d(
                    x=x_data, y=[name] * len(x_data), z=z_data[i], name=name,
                    mode="lines", line=dict(width=4), customdata=[name] * len(x_data)
                )
            )
        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis=dict(title="Time", range=[len(x_data) - 0.5, -0.5], autorange="reversed"),
                yaxis=dict(title="Day/Product"),
                zaxis=dict(title=value_title)
            ),
            clickmode="event+select", plot_bgcolor="#f9f9f9", margin=dict(l=0, r=0, b=0, t=40), showlegend=False
        )

    elif graph_type == "Horizon Chart":
        num_bands = 5
        pos_colors = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
        neg_colors = ["#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]

        max_abs = np.nanmax(np.abs(z_data)) if len(z_data) else 1
        if max_abs == 0 or np.isnan(max_abs):
            max_abs = 1

        band_size = max_abs / num_bands

        for i, name in enumerate(y_names):
            z = np.array(z_data[i], dtype=float)
            z_safe = np.nan_to_num(z, nan=0.0)
            custom_data_arr = [name] * len(x_data)

            fig.add_trace(go.Scatter(
                x=x_data, y=[i + 0.5] * len(x_data),
                mode="lines", line=dict(color='rgba(0,0,0,0)', width=1),
                name=name, customdata=custom_data_arr,
                text=[f"{val:.4f}" if not np.isnan(val) else "NaN" for val in z],
                hovertemplate="%{customdata} - Value: %{text}<extra></extra>",
                hoverinfo="all", showlegend=False
            ))

            for b in range(num_bands):
                pos_vals = np.clip(z_safe - b * band_size, 0, band_size)
                y_pos = i + (pos_vals / band_size)

                fig.add_trace(go.Scatter(
                    x=x_data, y=[i] * len(x_data), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                    customdata=custom_data_arr
                ))
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_pos, mode="lines", fill="tonexty", fillcolor=pos_colors[b],
                    line=dict(width=0.5, color=pos_colors[b]), showlegend=False, hoverinfo="skip",
                    customdata=custom_data_arr
                ))

                neg_vals = np.clip(-z_safe - b * band_size, 0, band_size)
                y_neg = i + (neg_vals / band_size)

                fig.add_trace(go.Scatter(
                    x=x_data, y=[i] * len(x_data), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                    customdata=custom_data_arr
                ))
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_neg, mode="lines", fill="tonexty", fillcolor=neg_colors[b],
                    line=dict(width=0.5, color=neg_colors[b]), showlegend=False, hoverinfo="skip",
                    customdata=custom_data_arr
                ))

        cscale = []
        for i in range(num_bands):
            idx = num_bands - 1 - i
            cscale.append([i / (2 * num_bands), neg_colors[idx]])
            cscale.append([(i + 1) / (2 * num_bands), neg_colors[idx]])
        for i in range(num_bands):
            cscale.append([(num_bands + i) / (2 * num_bands), pos_colors[i]])
            cscale.append([(num_bands + i + 1) / (2 * num_bands), pos_colors[i]])

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                colorscale=cscale, cmin=-max_abs, cmax=max_abs,
                showscale=True, colorbar=dict(title=value_title)
            ),
            showlegend=False, hoverinfo="skip"
        ))

        fig.update_layout(
            title=title_text,
            xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
            yaxis={
                "title": "Day/Product", "tickvals": list(range(len(y_names))),
                "ticktext": y_names, "range": [-0.5, len(y_names)]
            },
            clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9"
        )

    elif graph_type == "Spark Line":
        max_abs = np.nanmax(np.abs(z_data)) if len(z_data) else 1
        if max_abs == 0 or np.isnan(max_abs):
            max_abs = 1

        line_color_pos = "#1f77b4"
        fill_color_pos = "rgba(31, 119, 180, 0.4)"
        line_color_neg = "#d62728"
        fill_color_neg = "rgba(214, 39, 40, 0.4)"

        for i, name in enumerate(y_names):
            z = np.array(z_data[i], dtype=float)
            custom_data_arr = [name] * len(x_data)

            # 1. Invisible hover line (stays continuous for tooltips)
            fig.add_trace(go.Scatter(
                x=x_data, y=[i + 0.5] * len(x_data),
                mode="lines", line=dict(color='rgba(0,0,0,0)', width=1),
                name=name, customdata=custom_data_arr,
                text=[f"{val:.4f}" if not np.isnan(val) else "NaN" for val in z],
                hovertemplate="%{customdata} - Value: %{text}<extra></extra>",
                hoverinfo="all", showlegend=False
            ))

            # 2. Continuous faint grey baseline (keeps the visual row anchor intact)
            fig.add_trace(go.Scatter(
                x=x_data, y=[i] * len(x_data), mode="lines", line=dict(width=1, color="#dddddd"),
                showlegend=False, hoverinfo="skip", customdata=custom_data_arr
            ))

            # 3. HIDDEN baseline with NaNs inserted exactly where z has NaNs.
            baseline_gaps = [i if not np.isnan(val) else None for val in z]
            fig.add_trace(go.Scatter(
                x=x_data, y=baseline_gaps, mode="lines", line=dict(width=0, color='rgba(0,0,0,0)'),
                showlegend=False, hoverinfo="skip", customdata=custom_data_arr,
                connectgaps=False
            ))

            valid_mask = ~np.isnan(z)
            if np.any(valid_mask):
                z_max = np.max(z[valid_mask])
                z_min = np.min(z[valid_mask])

                # 4. Actual data line + fill POSITIVE
                if z_max > 0:
                    y_vals_pos = i + (np.where(np.isnan(z), np.nan, np.clip(z, 0, None)) / max_abs) * 0.45
                    fig.add_trace(go.Scatter(
                        x=x_data, y=y_vals_pos, mode="lines", fill="tonexty", fillcolor=fill_color_pos,
                        line=dict(width=1.5, color=line_color_pos), showlegend=False, hoverinfo="skip",
                        customdata=custom_data_arr,
                        connectgaps=False
                    ))

                # 5. Hidden baseline + gaps again for the negative fill (only add if we have negative data)
                if z_min < 0:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=baseline_gaps, mode="lines", line=dict(width=0, color='rgba(0,0,0,0)'),
                        showlegend=False, hoverinfo="skip", customdata=custom_data_arr,
                        connectgaps=False
                    ))

                    # 6. Actual data line + fill NEGATIVE
                    y_vals_neg = i + (np.where(np.isnan(z), np.nan, np.clip(z, None, 0)) / max_abs) * 0.45
                    fig.add_trace(go.Scatter(
                        x=x_data, y=y_vals_neg, mode="lines", fill="tonexty", fillcolor=fill_color_neg,
                        line=dict(width=1.5, color=line_color_neg), showlegend=False, hoverinfo="skip",
                        customdata=custom_data_arr,
                        connectgaps=False
                    ))

        fig.update_layout(
            title=title_text,
            xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
            yaxis={
                "title": "Day/Product", "tickvals": list(range(len(y_names))),
                "ticktext": y_names, "range": [-0.5, len(y_names)]
            },
            clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9"
        )

    else:  # Line Chart
        for i, name in enumerate(y_names):
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=z_data[i], name=name, mode="lines", customdata=[name] * len(x_data)
                )
            )
        fig.update_layout(
            title=title_text,
            xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
            yaxis={"title": value_title},
            showlegend=True,
            legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
            clickmode="event+select", hovermode="x unified", plot_bgcolor="#f9f9f9"
        )

    return fig


def main():
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    print("Loading data...")
    level_depth = 5
    all_data, names, detections = load_all_data(level_depth=level_depth)

    if not all_data:
        print("Warning: No data loaded. Make sure 'data/' has CSV files.")

    aggregated_data = aggregate_data(all_data, metric=metric, aggregation=aggregation_functions_map[chosen_aggregation],
                                     time_window=time_window_aggregation * MINUTE_SEC)
    print("Data loaded.")

    placeholder_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))
    placeholder_fig.update_layout(
        annotations=[{
            "text": "Select Day/Product using the dropdown above to view the price graph",
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
            "showarrow": False, "font": {"size": 20}
        }]
    )
    placeholder_fig.register_update_graph_callback(app=app, graph_id="price_graph")

    x_data_init = [f"{i // HOUR_SEC:02d}:{i % HOUR_SEC // MINUTE_SEC:02d}" for i in
                   range(0, DAY_SEC, time_window_aggregation * MINUTE_SEC)]

    data_processing_init = "normalize"
    if data_processing_init == "normalize":
        z_init = normalize_data(aggregated_data)
    elif data_processing_init == "pct_change":
        z_init = pct_change_data(aggregated_data)
    else:
        z_init = aggregated_data

    bottom_fig = create_bottom_figure(
        z_init,
        x_data_init,
        names,
        "Heatmap",
        metric,
        chosen_aggregation,
        data_processing_init
    )

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.P("Select LOB (Top Graph):", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="lob_selector",
                    options=[{"label": name, "value": name} for name in names],
                    value=names[0] if names else None,
                    clearable=False
                ),
            ], style={
                "marginBottom": "1rem",
                "padding": "0.5rem",
                "backgroundColor": "#ffffff",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
            }),
            dcc.Loading(
                dcc.Graph(
                    id="price_graph", figure=placeholder_fig,
                    config={"toImageButtonOptions": {"format": "png", "filename": "price_graph", "width": 1920,
                                                     "height": 1080, "scale": 3}}
                ), type="circle"
            )
        ], style={
            "marginBottom": "1rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
            "borderRadius": "10px", "padding": "0.5rem", "backgroundColor": "#f9f9f9"
        }),

        html.Div([
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
                    value=metric, clearable=False, style={"marginBottom": "0.5rem"}
                ),
                html.P(id="metric_description", style={"marginBottom": "1rem", "fontStyle": "italic"}),

                html.P("Select an aggregation function:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="aggregation_dropdown",
                    options=[{"label": x, "value": x} for x in ["Mean", "Median", "Max", "Min", "Std"]],
                    value=chosen_aggregation, clearable=False, style={"marginBottom": "1rem"}
                ),

                html.P("Select a time window for the bottom graph (in minutes) (1 - 720):",
                       style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Input(
                    id="time_window_input", type="number", value=time_window_aggregation, min=1, max=720, step=1,
                    placeholder="Time Window (minutes)", style={"width": "100%", "marginBottom": "1rem"}
                ),

                html.P("Data Processing:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="data_processing_dropdown",
                    options=[
                        {"label": "None (Raw Values)", "value": "none"},
                        {"label": "Normalize (Min-Max)", "value": "normalize"},
                        {"label": "Percentage Change (%)", "value": "pct_change"}
                    ],
                    value="normalize", clearable=False, style={"marginBottom": "1rem"}
                ),

                html.P("Select bottom graph style:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="graph_type_dropdown",
                    options=[
                        {"label": " Heatmap", "value": "Heatmap"},
                        {"label": " 2D Line chart", "value": "Line Chart"},
                        {"label": " 3D Line chart", "value": "3D Lines"},
                        {"label": " Horizon Chart", "value": "Horizon Chart"},
                        {"label": " Spark Line", "value": "Spark Line"}
                    ],
                    value="Heatmap", clearable=False, style={"marginBottom": "1rem"}
                ),

                html.P("Filter Time Series:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Dropdown(
                    id="timeseries_selector",
                    options=[{"label": name, "value": name} for name in names],
                    value=names.copy(),  # default = all selected
                    multi=True,
                    placeholder="Select time series...",
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
                    "fontSize": "1rem", "fontWeight": "bold", "color": "#fff", "backgroundColor": "#28a745",
                    "border": "none", "borderRadius": "8px", "boxShadow": "0 4px 6px rgba(40, 167, 69, 0.3)",
                    "cursor": "pointer", "transition": "background-color 0.3s ease-in-out, transform 0.2s ease-in-out",
                }),
            ], style={
                "flex": "0 0 300px", "padding": "0.5rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                "borderRadius": "10px", "backgroundColor": "#ffffff", "minWidth": "250px"
            }),

            html.Div([
                dcc.Loading(
                    dcc.Graph(
                        id="bottom_graph", figure=bottom_fig, clear_on_unhover=True,
                        config={"toImageButtonOptions": {"format": "png", "filename": "bottom_graph", "width": 1920,
                                                         "height": 1080, "scale": 3}}
                    ), type="circle"
                )
            ], style={
                "flex": "1", "marginLeft": "1rem", "padding": "0.5rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                "borderRadius": "10px", "backgroundColor": "#f9f9f9", "minWidth": "0"
            })
        ], style={"display": "flex", "flexWrap": "nowrap", "alignItems": "flex-start", "gap": "1rem"}),

        html.Div(id="dummy-clientside-output", style={"display": "none"})

    ], style={"padding": "0.5rem", "fontFamily": "Arial, sans-serif", "backgroundColor": "#f0f2f5"})

    @app.callback(
        Output("timeseries_selector", "value"),
        Input("select_all_btn", "n_clicks"),
        Input("unselect_all_btn", "n_clicks"),
        prevent_initial_call=True
    )
    def handle_select_buttons(select_all, unselect_all):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "select_all_btn":
            return names
        elif trigger == "unselect_all_btn":
            return []

        return dash.no_update

    @app.callback(
        Output("metric_description", "children"),
        Input("metric_dropdown", "value")
    )
    def update_description(selected_metric):
        return metric_descriptions_map[
            selected_metric] if selected_metric in metric_descriptions_map else "No description available."

    @app.callback(
        Output("price_graph", "figure"),
        Input("lob_selector", "value"),
    )
    def update_price_graph_from_dropdown(selected_name):
        if not selected_name:
            return dash.no_update

        idx = names.index(selected_name)
        data = all_data[idx]

        timestamps = data["Time"].values
        ask_prices = [data[f"Ask Price {i}"].values for i in range(1, 6)]
        bid_prices = [data[f"Bid Price {i}"].values for i in range(1, 6)]
        imbalance_indices = data["Imbalance Index"].values
        freqs = data["Frequency of Incoming Messages"].values
        cancels = data["Cancellations Rate"].values

        fig = create_price_graph(
            timestamps,
            ask_prices,
            bid_prices,
            imbalance_indices,
            freqs,
            cancels,
            selected_name,
            detections[idx]
        )

        fig.register_update_graph_callback(app=app, graph_id="price_graph")

        return fig

    @app.callback(
        Output("bottom_graph", "figure"),
        Input("update_bottom_graph_button", "n_clicks"),
        Input("graph_type_dropdown", "value"),
        Input("data_processing_dropdown", "value"),
        Input("timeseries_selector", "value"),
        State("metric_dropdown", "value"),
        State("aggregation_dropdown", "value"),
        State("time_window_input", "value"),
        State("bottom_graph", "figure"),
    )
    def update_bottom_graph(update_bottom_graph_button, selected_graph_type, data_processing,
                            selected_timeseries, selected_metric, selected_aggregation, selected_time_window,
                            bottom_fig):
        global last_update_bottom_graph_click_count, last_graph_type, last_data_processing_state
        updated = False

        ctx = dash.callback_context
        if not ctx.triggered: return dash.no_update
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id in ["update_bottom_graph_button", "graph_type_dropdown", "data_processing_dropdown",
                          "timeseries_selector"]:
            if trigger_id == "update_bottom_graph_button" and last_update_bottom_graph_click_count == update_bottom_graph_button and selected_graph_type == last_graph_type and data_processing == last_data_processing_state:
                pass
            else:
                last_update_bottom_graph_click_count = update_bottom_graph_button
                last_graph_type = selected_graph_type
                last_data_processing_state = data_processing

                selected_time_window_sec = selected_time_window * MINUTE_SEC
                new_x = [f"{int(i // HOUR_SEC):02d}:{int(i % HOUR_SEC // MINUTE_SEC):02d}" for i in
                         range(0, DAY_SEC, selected_time_window_sec)]
                new_z = aggregate_data(all_data, metric=selected_metric,
                                       aggregation=aggregation_functions_map[selected_aggregation],
                                       time_window=selected_time_window_sec)

                if data_processing == "normalize":
                    new_z = normalize_data(new_z)
                elif data_processing == "pct_change":
                    new_z = pct_change_data(new_z)

                # --- FILTERING ---
                if selected_timeseries:
                    selected_indices = [i for i, n in enumerate(names) if n in selected_timeseries]
                    filtered_names = [names[i] for i in selected_indices]
                    filtered_z = np.array(new_z)[selected_indices]
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

                bottom_fig = create_bottom_figure(
                    filtered_z,
                    new_x,
                    filtered_names,
                    selected_graph_type,
                    selected_metric,
                    selected_aggregation,
                    data_processing
                )

                if trigger_id != "update_bottom_graph_button" and current_range:
                    if selected_graph_type == "3D Lines":
                        bottom_fig.update_layout(scene=dict(xaxis={"range": current_range}))
                    else:
                        bottom_fig.update_layout(xaxis={"range": current_range})
                updated = True

        return bottom_fig if updated else dash.no_update

    app.clientside_callback(
        """
        function(figure) {
            setTimeout(function() {
                const graph = document.getElementById('bottom_graph');
                if (!graph) return;
                const plot = graph.querySelector('.js-plotly-plot');
                if (!plot) return;

                function attachLegendHover() {
                    const legendItems = plot.querySelectorAll('.legend .traces');
                    if (!legendItems.length) return;
                    legendItems.forEach((item, index) => {
                        item.onmouseenter = function() {
                            const traceCount = plot.data.length;
                            if (!plot._originalColors) plot._originalColors = plot.data.map(t => t.line?.color || null);
                            let update = { opacity: [], line: [] };
                            for (let i = 0; i < traceCount; i++) {
                                if (i === index) {
                                    update.opacity.push(1);
                                    update.line.push({ width: 5, color: plot._originalColors[i] });
                                } else {
                                    update.opacity.push(0.3);
                                    update.line.push({ width: 1.5, color: "rgba(180,180,180,0.7)" });
                                }
                            }
                            Plotly.restyle(plot, update);
                        };
                        item.onmouseleave = function() {
                            const traceCount = plot.data.length;
                            let update = { opacity: [], line: [] };
                            for (let i = 0; i < traceCount; i++) {
                                update.opacity.push(1);
                                update.line.push({ width: 2, color: plot._originalColors ? plot._originalColors[i] : null });
                            }
                            Plotly.restyle(plot, update);
                        };
                    });
                }
                attachLegendHover();
                if (!plot._legendHoverAttached) {
                    plot.on('plotly_afterplot', attachLegendHover);
                    plot._legendHoverAttached = true;
                }
            }, 300);
            return window.dash_clientside.no_update;
        }
        """,
        Output("dummy-clientside-output", "children"),
        Input("bottom_graph", "figure"),
        prevent_initial_call=False
    )

    app.run(host=HOST_ADDRESS, port=PORT, debug=False)


if __name__ == "__main__":
    main()