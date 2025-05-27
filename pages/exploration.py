import dash
from dash import html, dcc, callback, Input, Output, Dash, MATCH
import plotly.express as px
import pandas as pd
import io


dash.register_page(__name__)

layout = html.Div([
    # dcc.Store(id='dataset', storage_type='local'),
    html.Div([
        html.H1("Estadísticos clave"),
        html.Div(id="cards")
    ],
    id= 'Stats'),
    html.Div([
        html.H1("Correlación"),
        dcc.Graph(id='correlation'),
    ], id= 'corr'),
    html.H1("Histograms"),
    html.Div(id = 'histograms')
])

@callback(
    Output('cards', 'children'),
    Input('dataset', 'data')
)
def cards(df):
    df = pd.read_json(io.StringIO(df), orient='split')
    columns = df.columns
    print(columns)
    cards_list = []
    for col in columns:
        cards_list.append( html.Div([
            html.H1(col),
            html.P(f"Nulos: {df[col].isna().mean()*100}%"),
            html.P(f"Tipo: {df[col].dtype}")],
        className= 'card' ))
    print(f"cards_list: {len(cards_list)}")
    return cards_list

@callback(
    Output('correlation', 'figure'),
    Input('dataset', 'data')
)
def corr_matrix(df):
    df = pd.read_json(io.StringIO(df), orient='split')
    return px.imshow(
                df.dropna(axis='columns', thresh=0.3).dropna().corr(numeric_only=True),
                range_color=[-1, 1],
                color_continuous_scale='rdylgn')

@callback(
    Output('histograms', 'children'),
    Input('dataset', 'data')
)
def generate_histograms(json_data):
    if json_data is None:
        return html.Div("No data loaded.")

    df = pd.read_json(io.StringIO(json_data), orient='split')

    # Select numeric variables
    numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns

    # Generate a list of dcc.Graph components
    histograms = []
    for col in numeric_vars:
        fig = px.histogram(df, x=col, nbins=30, title=f'Histogram of {col}')
        histograms.append(
            dcc.Graph(
                figure=fig,
                config={'displayModeBar': False},
                className='histogram'
            )
        )

    if not histograms:
        return html.P("No numeric columns to display.")
    
    return histograms