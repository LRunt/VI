import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
import datetime

import plotly.graph_objects as go
import plotly_resampler
from plotly_resampler import FigureResampler

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from src.viz.dataloader import load_all_data, aggregate_data

# Networking settings for the Dash app
HOST_ADDRESS = "127.0.0.1"
PORT = 8080

# Minute in seconds (60 seconds)
MINUTE_SEC = 60
# Hour in seconds (60 minutes)
HOUR_SEC = 60 * MINUTE_SEC
# Day in seconds (24 hours)
DAY_SEC = 24 * HOUR_SEC

# Global variables to store graph settings across callbacks
timestamps = None  # Timestamps of the data
ask_prices = None  # Ask prices
bid_prices = None  # Bid prices

timestamps_graph_labels = None  # Timestamps for the graph

is_loading = False  # Flag to indicate if the data is loading
last_hover_label = None  # Last hovered label in the heatmap
last_heatmap_click_count = None  # Last clicked point in the heatmap
last_update_heatmap_click_count = None  # Last clicked "Apply" button in the heatmap
last_graph_type = "Heatmap"  # State tracker for graph type

chosen_aggregation = "Mean"  # Default aggregation function
aggregation_functions_map = {
    "Mean": np.mean,
    "Median": np.median,
    "Max": np.max,
    "Min": np.min,
    "Std": np.std
}
metric = "Ask Price 1"  # Default metric for aggregation
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
time_window_aggregation = 3600  # Default time window for aggregation (1 hour)


def create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels, name, detected_anomalies,
                       how_many_x_ticks=75):
    # Convert timestamps to HH:MM:SS format
    global timestamps_graph_labels
    timestamps_graph_labels = [datetime.datetime.fromtimestamp(int(ts) / 1e9 - HOUR_SEC).strftime("%H:%M:%S.%f") for ts
                               in timestamps]
    # Convert to 0 - n
    timestamps_graph = list(range(len(timestamps_graph_labels)))
    # Ticks
    tickvals = list(range(0, len(timestamps), len(timestamps) // how_many_x_ticks))
    ticklabels = [timestamps_graph_labels[i] for i in tickvals]
    # Go from HH:MM:SS.nnnnnn to truly HH:MM:SS
    ticklabels = [ts[:8] for ts in ticklabels]

    def interpolate_color(color1, color2, factor):
        return tuple(int(color1[i] + factor * (color2[i] - color1[i])) for i in range(3))

    # Price graph
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

        y_pred = np.pad(y_pred, (0, len(timestamps) - len(y_pred)), "constant", constant_values=1)
        y_pred = y_pred[:len(timestamps)]
        anomaly_proba = np.pad(anomaly_proba, (0, len(timestamps) - len(anomaly_proba)), "constant", constant_values=0)
        anomaly_proba = anomaly_proba[:len(timestamps)]

        anomaly_alpha = anomaly_proba[y_pred == -1]
        anomaly_timestamps = [timestamps_graph[i] for i in range(len(y_pred)) if y_pred[i] == -1]

        anomaly_alpha = (anomaly_alpha - anomaly_alpha.min()) / (anomaly_alpha.max() - anomaly_alpha.min())
        anomaly_alpha = np.nan_to_num(anomaly_alpha)

        if len(timestamps) > 500_000:
            anomaly_alpha *= 0.1
        if len(timestamps) > 1_000_000:
            anomaly_alpha *= 0.5

        prices = np.array(bid_prices + ask_prices)
        y_min = np.nanmin(prices)
        y_max = np.nanmax(prices)

        x_vals = []
        x_vals_with_Nones = []
        y_vals = []
        y_vals_with_Nones = []

        for i, ts in enumerate(anomaly_timestamps):
            x_vals.extend([ts, ts])
            y_vals.extend([y_min, y_max])
            x_vals_with_Nones.extend([ts, ts, None])
            y_vals_with_Nones.extend([y_min, y_max, None])

        try:
            price_graph_fig.add_trace(
                go.Scattergl(name="Detected Anomaly", yaxis="y1", mode="lines",
                             line=dict(width=1, color="rgba(0, 0, 0, 0.75)"), hoverinfo="skip"),
                hf_x=x_vals_with_Nones, hf_y=y_vals_with_Nones,
            )
        except Exception as e:
            price_graph_fig.add_trace(
                go.Scattergl(name="Detected Anomaly", yaxis="y1", mode="lines",
                             line=dict(width=1, color="rgba(0, 0, 0, 0.75)"), hoverinfo="skip"),
                hf_x=x_vals, hf_y=y_vals,
            )

    price_graph_fig.add_trace(
        go.Scattergl(
            name="Highlight", yaxis="y1", mode="lines", fill="toself",
            line=dict(width=2, color="rgba(25, 25, 100, 1)"), fillcolor="rgba(185, 215, 255, 0.3)",
            hoverinfo="skip", showlegend=False,
        ),
        hf_x=[], hf_y=[],
    )

    price_graph_fig.update_layout(
        title=f"{name}",
        xaxis={"title": "Timestamp", "tickmode": "array", "tickvals": tickvals, "ticktext": ticklabels,
               "range": [0, len(timestamps_graph_labels) - 1]},
        yaxis={"title": "Price", "side": "left"},
        yaxis2={"title": "Imbalance index", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True,
                "range": [-1, 1]},
        yaxis3={"title": "Incoming messages (per sec)", "side": "right", "overlaying": "y", "anchor": "free",
                "autoshift": True},
        yaxis4={"title": "Cancellations rate", "side": "right", "overlaying": "y", "anchor": "free", "autoshift": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        clickmode="event+select",
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
    )

    for trace in price_graph_fig.data:
        trace.name = trace.name.split("~")[0].strip()
        trace.name = trace.name.replace("[R]", "").strip()

    return price_graph_fig


def create_bottom_figure(z_data, x_data, y_names, graph_type, metric_name, agg_name):
    """
    Helper function to dynamically build either a Heatmap or a Line Chart
    """
    fig = go.Figure()

    if graph_type == "Heatmap":
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_data,
                y=y_names,
                colorscale="Viridis",
                colorbar=dict(),
                hoverongaps=False,
                zmin=np.min(z_data),
                zmax=np.max(z_data),
            )
        )
        fig.update_layout(yaxis={"title": "Day/Product"})
    else:  # Line Chart
        for i, name in enumerate(y_names):
            fig.add_trace(
                go.Scattergl(
                    x=x_data,
                    y=z_data[i],
                    name=name,
                    mode="lines",
                    # Store the name in customdata to robustly identify it on click
                    customdata=[name] * len(x_data)
                )
            )
        fig.update_layout(
            yaxis={"title": "Normalized Value"},
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5}
        )

    fig.update_layout(
        title=f"{agg_name} of {metric_name} ({graph_type}) - Normalized Values",
        xaxis={"title": "Time", "range": [-0.5, len(x_data) - 0.5]},
        clickmode="event+select",
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
    )
    return fig


def main():
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    print("Loading data...")
    level_depth = 5
    all_data, names, detections = load_all_data(level_depth=level_depth)
    aggregated_data = aggregate_data(all_data, metric=metric, aggregation=aggregation_functions_map[chosen_aggregation],
                                     time_window=time_window_aggregation)
    print("Data loaded.")

    # Placeholder for the price graph
    placeholder_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))
    placeholder_fig.update_layout(
        annotations=[
            {
                "text": "Select Day/Product in the bottom graph to view the price graph",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "showarrow": False,
                "font": {"size": 20}
            }
        ]
    )
    placeholder_fig.register_update_graph_callback(app=app, graph_id="price_graph")

    # Initial Bottom Graph creation (Heatmap default)
    x_data_init = [f"{i // HOUR_SEC:02d}:{i % HOUR_SEC // MINUTE_SEC:02d}" for i in
                   range(0, DAY_SEC, time_window_aggregation)]
    bottom_fig = create_bottom_figure(aggregated_data, x_data_init, names, "Heatmap", metric, chosen_aggregation)

    # HTML Layout
    app.layout = html.Div([
        # Price Graph at the top
        html.Div([
            dcc.Loading(
                dcc.Graph(
                    id="price_graph",
                    figure=placeholder_fig,
                    config={
                        "toImageButtonOptions": {
                            "format": "png", "filename": "price_graph",
                            "width": 1920, "height": 1080, "scale": 3
                        }
                    }
                ),
                type="circle"
            )
        ], style={
            "marginBottom": "1rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
            "borderRadius": "10px", "padding": "0.5rem", "backgroundColor": "#f9f9f9"
        }),

        # Heatmap/Linechart + Settings
        html.Div([
            # Settings (on the left)
            html.Div([
                html.H4("Bottom Graph Settings", style={"marginBottom": "1rem"}),

                # NEW: Graph type selector
                html.P("Select graph style:", style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.RadioItems(
                    id="graph_type_radio",
                    options=[
                        {"label": " Heatmap", "value": "Heatmap"},
                        {"label": " Line Chart", "value": "Line Chart"}
                    ],
                    value="Heatmap",
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"display": "inline-block", "marginRight": "15px"},
                    style={"marginBottom": "1rem"}
                ),
                html.Hr(),

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

                html.P("Select a time window (seconds) (60 - 43200):",
                       style={"fontWeight": "bold", "marginBottom": "0.5rem"}),
                dcc.Input(
                    id="time_window_input", type="number", value=time_window_aggregation,
                    min=60, max=43200, step=1, placeholder="Time Window (seconds)",
                    style={"width": "100%", "marginBottom": "1rem"}
                ),

                html.Button("Apply", id="update_heatmap_button", n_clicks=0, style={
                    "marginTop": "1rem", "width": "100%", "padding": "0.75rem 1rem",
                    "fontSize": "1rem", "fontWeight": "bold", "color": "#fff",
                    "backgroundColor": "#28a745", "border": "none", "borderRadius": "8px",
                    "boxShadow": "0 4px 6px rgba(40, 167, 69, 0.3)", "cursor": "pointer",
                    "transition": "background-color 0.3s ease-in-out, transform 0.2s ease-in-out",
                }),
            ], style={
                "flex": "0 0 300px", "padding": "0.5rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                "borderRadius": "10px", "backgroundColor": "#ffffff", "minWidth": "250px"
            }),

            # Bottom Graph (on the right)
            html.Div([
                dcc.Loading(
                    dcc.Graph(
                        id="heatmap_graph",  # Keeping ID same to avoid changing everything
                        figure=bottom_fig,
                        clear_on_unhover=True,
                        config={
                            "toImageButtonOptions": {
                                "format": "png", "filename": "bottom_graph",
                                "width": 1920, "height": 1080, "scale": 3
                            }
                        }
                    ),
                    type="circle"
                )
            ], style={
                "flex": "1", "marginLeft": "1rem", "padding": "0.5rem",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "borderRadius": "10px",
                "backgroundColor": "#f9f9f9", "minWidth": "0"
            })
        ], style={
            "display": "flex", "flexWrap": "nowrap", "alignItems": "flex-start", "gap": "1rem",
        }),
    ], style={"padding": "0.5rem", "fontFamily": "Arial, sans-serif", "backgroundColor": "#f0f2f5"})

    @app.callback(
        Output("metric_description", "children"),
        Input("metric_dropdown", "value")
    )
    def update_description(selected_metric):
        return metric_descriptions_map[
            selected_metric] if selected_metric in metric_descriptions_map else "No description available."

    @app.callback(
        Output("price_graph", "figure"),
        Input("heatmap_graph", "clickData"),
    )
    def update_price_graph(heatmap_click):
        global timestamps, ask_prices, bid_prices, last_heatmap_click_count, last_hover_label, is_loading

        if heatmap_click and last_heatmap_click_count != heatmap_click:
            is_loading = True
            last_heatmap_click_count = heatmap_click
            last_hover_label = None

            clicked_point = heatmap_click["points"][0]

            # Zjištění, o jaký produkt se jedná nezávisle na tom, jaký graf je dole
            if "customdata" in clicked_point:
                # Uživatelské kliknutí do Line Chartu
                clicked_name = clicked_point["customdata"]
            else:
                # Uživatelské kliknutí do Heatmapy
                clicked_name = clicked_point["y"]

            clicked_index = names.index(clicked_name)
            clicked_data = all_data[clicked_index]

            timestamps = clicked_data["Time"].values
            ask_prices = [clicked_data[f"Ask Price {i}"].values for i in range(1, level_depth + 1)]
            bid_prices = [clicked_data[f"Bid Price {i}"].values for i in range(1, level_depth + 1)]
            imbalance_indices = clicked_data["Imbalance Index"].values
            freqs = clicked_data["Frequency of Incoming Messages"].values
            cancels = clicked_data["Cancellations Rate"].values

            price_fig = create_price_graph(timestamps, ask_prices, bid_prices, imbalance_indices, freqs, cancels,
                                           names[clicked_index], detections[clicked_index])
            price_fig.register_update_graph_callback(app=app, graph_id="price_graph")

            is_loading = False
            return price_fig

        return dash.no_update

    @app.callback(
        Output("price_graph", "figure", allow_duplicate=True),
        Input("heatmap_graph", "hoverData"),
        State("time_window_input", "value"),
        State("price_graph", "figure"),
        prevent_initial_call=True,
    )
    def update_highlight(heatmap_hover, selected_time_window, price_fig):
        global last_hover_label
        if is_loading:
            return dash.no_update

        updated = False
        if isinstance(price_fig, dict):
            price_fig = FigureResampler(price_fig, default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))

        if heatmap_hover and last_hover_label != heatmap_hover:
            updated = True
            hovered_label = heatmap_hover["points"][0]["x"]
            last_hover_label = hovered_label

            h, m = map(int, hovered_label.split(":"))
            hovered_sec = h * HOUR_SEC + m * MINUTE_SEC

            highlight_start = hovered_sec
            highlight_end = hovered_sec + selected_time_window

            def find_index_for_sec(sec):
                global timestamps
                if timestamps is None:
                    return 0
                timestamps_series = pd.Series(pd.to_datetime(timestamps, unit="ns"))
                seconds_since_midnight = (timestamps_series - timestamps_series.dt.normalize()).dt.total_seconds()
                for i, ts in enumerate(seconds_since_midnight):
                    if ts >= sec:
                        return i
                return len(seconds_since_midnight) - 1

            x0 = find_index_for_sec(highlight_start)
            x1 = find_index_for_sec(highlight_end)

            if bid_prices is None or ask_prices is None:
                prices = np.array([0, 1])
            else:
                prices = np.array(bid_prices + ask_prices)
            y_min = np.nanmin(prices)
            y_max = np.nanmax(prices)

            highlight_x = [x0, x1, x1, x0, x0]
            highlight_y = [y_max, y_max, y_min, y_min, y_max]

            current_range = price_fig.layout["xaxis"]["range"] if "xaxis" in price_fig.layout and "range" in \
                                                                  price_fig.layout["xaxis"] else None
            price_fig.update_traces(
                selector=dict(name="Highlight"), x=highlight_x, y=highlight_y,
            )
            if current_range:
                price_fig.update_layout(xaxis={"range": current_range})

        if heatmap_hover is None and last_hover_label is not None:
            updated = True
            last_hover_label = None
            price_fig.update_traces(selector=dict(name="Highlight"), x=[], y=[])

        if updated:
            for trace in price_fig.data:
                trace.name = trace.name.split("~")[0].strip()
                trace.name = trace.name.replace("[R]", "").strip()

        return price_fig if updated else dash.no_update

    @app.callback(
        Output("heatmap_graph", "figure"),
        Input("update_heatmap_button", "n_clicks"),
        Input("price_graph", "relayoutData"),
        Input("graph_type_radio", "value"),
        State("metric_dropdown", "value"),
        State("aggregation_dropdown", "value"),
        State("time_window_input", "value"),
        State("heatmap_graph", "figure"),
    )
    def update_heatmap(update_heatmap_button, price_relayout, selected_graph_type, selected_metric,
                       selected_aggregation, selected_time_window, heatmap_fig):
        global last_update_heatmap_click_count
        global last_graph_type

        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        updated = False

        if isinstance(heatmap_fig, dict):
            heatmap_fig = go.Figure(heatmap_fig)

        # 1. Řešíme změnu metriky nebo typu grafu
        if trigger_id in ["update_heatmap_button", "graph_type_radio"]:
            if trigger_id == "update_heatmap_button" and last_update_heatmap_click_count == update_heatmap_button and selected_graph_type == last_graph_type:
                pass  # Žádná skutečná změna
            else:
                last_update_heatmap_click_count = update_heatmap_button
                last_graph_type = selected_graph_type

                new_x = [f"{int(i // HOUR_SEC):02d}:{int(i % HOUR_SEC // MINUTE_SEC):02d}" for i in
                         range(0, DAY_SEC, selected_time_window)]
                new_z = aggregate_data(all_data, metric=selected_metric,
                                       aggregation=aggregation_functions_map[selected_aggregation],
                                       time_window=selected_time_window)

                # Uložení existujícího zoomu pro zachování pozice při změně typu grafu
                current_range = None
                if heatmap_fig and "layout" in heatmap_fig and "xaxis" in heatmap_fig.layout and "range" in heatmap_fig.layout.xaxis:
                    current_range = heatmap_fig.layout.xaxis.range

                heatmap_fig = create_bottom_figure(new_z, new_x, names, selected_graph_type, selected_metric,
                                                   selected_aggregation)

                # Aplikování starého zoomu, pokud nějaký byl
                if current_range:
                    heatmap_fig.update_layout(xaxis={"range": current_range})

                updated = True

        # 2. Řešíme zoom z hlavního grafu
        if trigger_id == "price_graph" and price_relayout and timestamps_graph_labels:
            updated = True

            x0 = price_relayout.get("xaxis.range[0]", 0)
            x1 = price_relayout.get("xaxis.range[1]", len(timestamps_graph_labels) - 1)

            x0 = max(0, min(x0, len(timestamps_graph_labels) - 1))
            x1 = max(0, min(x1, len(timestamps_graph_labels) - 1))

            t0 = timestamps_graph_labels[int(x0)]
            t1 = timestamps_graph_labels[int(x1)]

            t0_parts = t0.split(":")
            t1_parts = t1.split(":")
            hour0, minute0, second0 = int(t0_parts[0]), int(t0_parts[1]), int(t0_parts[2].split(".")[0])
            hour1, minute1, second1 = int(t1_parts[0]), int(t1_parts[1]), int(t1_parts[2].split(".")[0])
            t0_sec = hour0 * HOUR_SEC + minute0 * MINUTE_SEC + second0
            t1_sec = hour1 * HOUR_SEC + minute1 * MINUTE_SEC + second1

            heatmap_x = heatmap_fig.data[0]["x"]
            heatmap_sec = [int(x.split(":")[0]) * HOUR_SEC + int(x.split(":")[1]) * MINUTE_SEC for x in heatmap_x]

            t0_index = max(i for i, sec in enumerate(heatmap_sec) if sec <= t0_sec)
            t1_index = min(i for i, sec in enumerate(heatmap_sec) if sec >= t1_sec)
            t0_index = round(max(0, min(t0_index, len(heatmap_sec) - 1)) - 0.5, 3)
            t1_index = round(max(0, min(t1_index, len(heatmap_sec) - 1)) - 0.5, 3)

            heatmap_fig.update_layout(xaxis={"range": [t0_index, t1_index]})

        return heatmap_fig if updated else dash.no_update

    app.run(host=HOST_ADDRESS, port=PORT, debug=False)


if __name__ == "__main__":
    main()