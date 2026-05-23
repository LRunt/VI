import os
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly_resampler
from plotly_resampler import FigureResampler

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Pokročilé analytické knihovny
from sklearn.cluster import KMeans
import umap.umap_ as umap
import networkx as nx
import networkx.algorithms.community as nx_comm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

HOST_ADDRESS = "127.0.0.1"
PORT = 8080

df_raw = None
stock_names = None

def load_data(filepath="intraday_stocks_staggered.csv"):
    """Načte vygenerovaná intradenní data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Soubor {filepath} nebyl nalezen. Nejdříve vygeneruj data.")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    names = df.columns.tolist()
    return df, names

def create_price_graph(stock_name, df_stock):
    """Vykreslí detailní graf pro jednu vybranou akcii."""
    price_graph_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))
    price_graph_fig.add_trace(
        go.Scattergl(
            name=stock_name,
            mode='lines',
            line=dict(color='#007bff', width=2),
            connectgaps=False
        ),
        hf_x=df_stock.index,
        hf_y=df_stock.values
    )
    price_graph_fig.update_layout(
        title=f"Detailní pohled: {stock_name}",
        xaxis={"title": "Čas", "autorange": True},
        yaxis={"title": "Cena (USD)", "autorange": True},
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return price_graph_fig

def create_bottom_figure(df_resampled, graph_type):
    """Vykreslí přehledový graf trhu podle zvolené analytické metody."""
    fig = go.Figure()
    
    x_data = df_resampled.index
    y_names = df_resampled.columns
    
    df_filled = df_resampled.ffill().fillna(0)
    df_norm = (df_filled - df_filled.min()) / (df_filled.max() - df_filled.min())
    df_norm = df_norm.fillna(0)
    X = df_norm.T.values 

    # --- UMAP PROJEKCE ---
    if graph_type == "UMAP Projekce (Scatter)":
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        embedding = reducer.fit_transform(X)
        
        fig = px.scatter(
            x=embedding[:, 0], y=embedding[:, 1], 
            color=[str(L) for L in labels],
            hover_name=y_names, labels={"color": "Shluk ID"}
        )
        fig.update_traces(customdata=y_names)
        fig.update_layout(title="UMAP Projekce: Blízké body mají podobný vývoj", plot_bgcolor="#f9f9f9", clickmode="event+select", uirevision="constant")

    # --- 2D LINECHART (CENTROIDY) ---
    elif graph_type == "2D Linechart (Centroidy)":
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        
        colors = px.colors.qualitative.Plotly
        for i in range(n_clusters):
            cluster_size = sum(labels == i)
            fig.add_trace(go.Scattergl(x=x_data, y=centroids[i], name=f"Shluk {i} ({cluster_size} akcií)", mode="lines", line=dict(width=4, color=colors[i % len(colors)]), customdata=[f"Cluster_{i}"] * len(x_data)))
        fig.update_layout(title=f"Přehled trhu: Průměrné trendy {n_clusters} shluků", hovermode="x unified", plot_bgcolor="#f9f9f9", clickmode="event+select", uirevision="constant")

    # --- KORELAČNÍ SÍŤ S MINIMÁLNÍ KOSTROU ---
    elif graph_type == "Korelační síť (Čistá páteř)":
        corr_matrix = df_resampled.corr()
        G = nx.Graph()
        threshold = 0.80
        
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and corr_val > threshold:
                    G.add_edge(cols[i], cols[j], weight=corr_val)
                    
        if len(G.nodes()) > 0:
            G = nx.maximum_spanning_tree(G)
            G.remove_nodes_from(list(nx.isolates(G)))

        if len(G.nodes()) == 0:
            fig.update_layout(title=f"Korelační síť: Žádné vazby nad {threshold}", plot_bgcolor="#f9f9f9")
            return fig
            
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]][0], pos[edge[0]][1]
            x1, y1 = pos[edge[1]][0], pos[edge[1]][1]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines', showlegend=False))

        communities = list(nx_comm.greedy_modularity_communities(G))
        node_comm_map = {node: i for i, comm in enumerate(communities) for node in comm}

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [node_comm_map.get(node, 0) for node in G.nodes()] 
        node_adjacencies = [len(list(G.neighbors(node))) for node in G.nodes()]
        node_text = [f"{node}<br>Komunita: {node_comm_map.get(node, 0)}<br>Spojení: {adj}" for node, adj in zip(G.nodes(), node_adjacencies)]
        
        fig.add_trace(
            go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, customdata=list(G.nodes()), 
                marker=dict(showscale=False, colorscale='Plotly3', color=node_colors, size=[12 + (adj * 3) for adj in node_adjacencies], line_width=1.5, line_color="white"))
        )
        fig.update_layout(title=f"Čistá páteřní síť (Korelace > {threshold})", showlegend=False, hovermode='closest', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor="#f9f9f9", clickmode="event+select", uirevision="constant")

    # --- SHLUKOVANÁ KORELAČNÍ MATICE ---
    elif graph_type == "Korelační matice (Shlukovaná)":
        corr_matrix = df_resampled.corr().fillna(0)
        dist_array = pdist(corr_matrix.values, metric='euclidean')
        linkage_matrix = linkage(dist_array, method='ward')
        ordered_indices = leaves_list(linkage_matrix)
        ordered_cols = corr_matrix.columns[ordered_indices]
        ordered_corr = corr_matrix.loc[ordered_cols, ordered_cols]
        
        fig.add_trace(go.Heatmap(z=ordered_corr.values, x=ordered_cols, y=ordered_cols, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, hoverongaps=False))
        fig.update_layout(title="Shlukovaná korelační matice", xaxis={"showticklabels": False}, yaxis={"showticklabels": False, "autorange": "reversed"}, plot_bgcolor="#f9f9f9", clickmode="event+select", uirevision="constant")

    # --- NOVÉ: MALÉ NÁSOBKY (SPARKLINES) ---
    elif graph_type == "Malé násobky (Top korelující)":
        # Vybereme první akcii z dat jako referenční cíl
        target_stock = y_names[0]
        # Spočítáme korelaci všech vůči cíli
        corr = df_resampled.corr()[target_stock].sort_values(ascending=False).dropna()
        # Vybereme 16 nejlepších
        top_stocks = corr.index[:16]
        
        fig = make_subplots(rows=4, cols=4, subplot_titles=[f"{s} (r={corr[s]:.2f})" for s in top_stocks], vertical_spacing=0.08, horizontal_spacing=0.05)
        
        for i, stock in enumerate(top_stocks):
            row = (i // 4) + 1
            col = (i % 4) + 1
            # Normalizujeme pro srovnatelnost tvarů
            norm_y = (df_resampled[stock] - df_resampled[stock].min()) / (df_resampled[stock].max() - df_resampled[stock].min())
            color = 'blue' if stock == target_stock else 'green'
            
            fig.add_trace(go.Scattergl(x=x_data, y=norm_y, mode='lines', line=dict(width=1.5, color=color), customdata=[stock]*len(x_data)), row=row, col=col)
            
            # Vypneme osy u každého subgrafu pro čistotu (jiskrové grafy)
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
            
        fig.update_layout(title=f"Malé násobky: 16 akcií s nejvyšší korelací vůči referenční {target_stock}", showlegend=False, plot_bgcolor="#f9f9f9", clickmode="event+select", uirevision="constant")

    # --- NOVÉ: RIDGELINE / HORIZON ---
    elif graph_type == "Ridgeline (Top 20 volatilních)":
        # Vybereme 20 akcií, které se nejvíce hýbou
        top_20 = df_resampled.var().nlargest(20).index
        colors = px.colors.qualitative.Alphabet
        
        for i, stock in enumerate(top_20[::-1]): # Kreslíme odzadu, aby se překrývaly správně zespoda
            # Min-Max Normalizace
            min_val = df_resampled[stock].min()
            max_val = df_resampled[stock].max()
            norm_price = (df_resampled[stock] - min_val) / (max_val - min_val)
            
            # Klíčový trik Horizon/Ridgeline grafu - umělý posun na ose Y
            shifted_y = norm_price + (i * 0.3) 
            
            fig.add_trace(go.Scattergl(
                x=x_data, y=shifted_y, mode='lines', fill='tozeroy', 
                name=stock, line=dict(color=colors[i % len(colors)], width=1), 
                customdata=[stock]*len(x_data)
            ))
            
        fig.update_layout(
            title="Ridgeline Graf: Vývoj 20 nejvolatilnějších akcií přes sebe", 
            showlegend=False, plot_bgcolor="#f9f9f9", clickmode="event+select", uirevision="constant",
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

    # --- NOVÉ: PARALELNÍ SOUŘADNICE ---
    elif graph_type == "Paralelní souřadnice (Rysy)":
        # 1. Extrakce statistických vlastností (Features)
        # Výnos za dané období
        returns = (df_resampled.iloc[-1] - df_resampled.iloc[0]) / df_resampled.iloc[0]
        # Volatilita (směrodatná odchylka procentuálních změn)
        volatility = df_resampled.pct_change().std()
        # Maximální propad (Max Drawdown)
        max_dd = (df_resampled / df_resampled.cummax() - 1).min()
        
        features = pd.DataFrame({'Výnos': returns, 'Volatilita': volatility, 'Max Drawdown': max_dd}).dropna()
        
        # Plotly Parcoords pracují primárně s číselnými id
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=features['Volatilita'], colorscale='Plasma', showscale=True, cmin=features['Volatilita'].min(), cmax=features['Volatilita'].max()),
            dimensions=[
                dict(label='Max Drawdown (Riziko)', values=features['Max Drawdown']),
                dict(label='Volatilita (Denní výkyvy)', values=features['Volatilita']),
                dict(label='Celkový Výnos (Zisk)', values=features['Výnos'])
            ]
        ))
        fig.update_layout(title="Paralelní souřadnice: Filtr statistických vlastností. Vyznačte myší oblasti na osách.", plot_bgcolor="#f9f9f9", uirevision="constant")

    # --- PŮVODNÍ 3D LINECHART ---
    else:  
        z_data_raw_norm = ((df_resampled - df_resampled.min()) / (df_resampled.max() - df_resampled.min())).T.values
        for i, name in enumerate(y_names):
            if np.isnan(z_data_raw_norm[i]).all(): continue
            fig.add_trace(go.Scatter3d(x=x_data, y=[name] * len(x_data), z=z_data_raw_norm[i], name=name, mode="lines", line=dict(width=2), connectgaps=False, customdata=[name] * len(x_data)))
        fig.update_layout(title="Přehled trhu (3D Linechart)", scene=dict(xaxis_title="Čas", yaxis_title="Akcie", zaxis_title="Norm. cena"), margin=dict(l=0, r=0, b=0, t=40), showlegend=False, clickmode="event+select", uirevision="constant")

    return fig


def main():
    global df_raw, stock_names
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    print("Načítám generovaná data...")
    df_raw, stock_names = load_data("intraday_stocks_staggered.csv")
    
    placeholder_fig = FigureResampler(go.Figure())
    placeholder_fig.update_layout(
        annotations=[{"text": "Vyber akcii, blok nebo centroid ve spodním grafu", 
                      "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}],
        xaxis={"visible": False}, yaxis={"visible": False}
    )

    df_initial = df_raw.resample('10min').last()
    bottom_fig_init = create_bottom_figure(df_initial, "Korelační matice (Shlukovaná)")

    app.layout = html.Div([
        html.Div([dcc.Loading(dcc.Graph(id="price_graph", figure=placeholder_fig), type="circle")], style={"marginBottom": "1rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "padding": "0.5rem", "backgroundColor": "#fff", "borderRadius": "10px"}),
        html.Div([
            html.Div([
                html.P("Jemnost dat (v min):", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="granularity_dropdown", options=[1, 5, 10, 15, 30, 60], value=10, clearable=False, style={"marginBottom": "1.5rem"}),
                html.P("Typ vizualizace:", style={"fontWeight": "bold"}),
                dcc.RadioItems(
                    id="graph_type_radio",
                    options=[
                        {"label": " Korelační matice (Shlukovaná)", "value": "Korelační matice (Shlukovaná)"},
                        {"label": " Korelační síť (Čistá páteř)", "value": "Korelační síť (Čistá páteř)"},
                        {"label": " Malé násobky (Top korelující)", "value": "Malé násobky (Top korelující)"},
                        {"label": " Ridgeline (Top 20 volatilních)", "value": "Ridgeline (Top 20 volatilních)"},
                        {"label": " Paralelní souřadnice (Rysy)", "value": "Paralelní souřadnice (Rysy)"},
                        {"label": " UMAP Projekce (Scatter)", "value": "UMAP Projekce (Scatter)"},
                        {"label": " 2D Linechart (Centroidy)", "value": "2D Linechart (Centroidy)"},
                        {"label": " 3D Linechart", "value": "3D Linechart"}
                    ],
                    value="Korelační matice (Shlukovaná)",
                    labelStyle={"display": "block", "marginBottom": "5px"},
                    style={"marginBottom": "1rem"}
                ),
                html.Button("Aplikovat", id="update_bottom_graph_button", n_clicks=0, style={"width": "100%", "padding": "0.75rem", "backgroundColor": "#28a745", "color": "#fff", "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontWeight": "bold"})
            ], style={"flex": "0 0 280px", "padding": "1rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "backgroundColor": "#fff", "borderRadius": "10px"}),
            
            html.Div([
                dcc.Loading(
                    dcc.Graph(
                        id="bottom_graph", 
                        figure=bottom_fig_init, 
                        clear_on_unhover=True, 
                        responsive=True,
                        style={"height": "75vh", "width": "100%"}
                    ), 
                    type="circle"
                )
            ], style={"flex": "1", "marginLeft": "1rem", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "backgroundColor": "#fff", "borderRadius": "10px", "display": "flex", "flexDirection": "column"})
        ], style={"display": "flex"})
    ], style={"padding": "1rem", "backgroundColor": "#f0f2f5", "fontFamily": "Arial, sans-serif"})

    @app.callback(
        Output("price_graph", "figure"),
        Input("bottom_graph", "clickData"),
        State("granularity_dropdown", "value"),
        prevent_initial_call=True
    )
    def update_price_graph(clickData, granularity):
        if not clickData: return dash.no_update
        clicked_point = clickData["points"][0]
        
        if "x" in clicked_point and isinstance(clicked_point["x"], str) and clicked_point["x"].startswith("STOCK"):
            identifier = clicked_point["x"]
        else:
            identifier = clicked_point.get("customdata", clicked_point.get("y", None))
            
        if not identifier: return dash.no_update
        
        price_graph_fig = FigureResampler(go.Figure(), default_downsampler=plotly_resampler.MinMaxLTTB(parallel=True))
        
        if str(identifier).startswith("Cluster_"):
            cluster_id = int(identifier.split("_")[1])
            df_resampled = df_raw.resample(f'{granularity}min').last()
            df_filled = df_resampled.ffill().fillna(0)
            df_norm = (df_filled - df_filled.min()) / (df_filled.max() - df_filled.min())
            df_norm = df_norm.fillna(0)
            X = df_norm.T.values
            
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            cluster_stocks = np.array(df_norm.columns)[labels == cluster_id]
            
            for stock in cluster_stocks:
                df_stock = df_raw[stock].dropna()
                if not df_stock.empty:
                    price_graph_fig.add_trace(
                        go.Scattergl(x=df_stock.index, y=df_stock.values, mode='lines', line=dict(width=1.5, color='rgba(0, 123, 255, 0.4)'), name=stock, connectgaps=False, hoverinfo='skip'),
                        hf_x=df_stock.index, hf_y=df_stock.values
                    )
            
            price_graph_fig.update_layout(
                title=f"Detailní pohled: Všechny akcie ve Shluku {cluster_id} (Celkem {len(cluster_stocks)} akcií)",
                xaxis={"title": "Čas", "autorange": True}, yaxis={"title": "Cena (USD)", "autorange": True},
                plot_bgcolor="#f9f9f9", margin=dict(l=40, r=40, t=40, b=40), showlegend=False
            )
            return price_graph_fig

        else:
            stock_name = str(identifier)
            if stock_name in df_raw.columns:
                df_stock = df_raw[stock_name]
                valid_idx = df_stock.dropna().index
                if len(valid_idx) > 0:
                    df_stock = df_stock.loc[valid_idx[0]:valid_idx[-1]]
                return create_price_graph(stock_name, df_stock)
            return dash.no_update

    @app.callback(
        Output("bottom_graph", "figure"),
        Input("update_bottom_graph_button", "n_clicks"),
        State("granularity_dropdown", "value"),
        State("graph_type_radio", "value"),
        prevent_initial_call=True
    )
    def update_bottom_graph(n_clicks, granularity, graph_type):
        df_resampled = df_raw.resample(f'{granularity}min').last()
        return create_bottom_figure(df_resampled, graph_type)

    app.run(host=HOST_ADDRESS, port=PORT, debug=False)

if __name__ == "__main__":
    main()