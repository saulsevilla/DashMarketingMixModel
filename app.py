import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)
app.title = "MMM Dashboard"

app.layout = html.Div([
    dcc.Store(id='dataset', storage_type='local', data={}),
    dcc.Store(id='model', storage_type='local', data={}),
    dcc.Store(id='theme', storage_type='local', data={}),
    html.Header([
        html.H1('Multi-page app with Dash Pages'),
        dcc.Checklist(
            id='theme-toggle',
            options=[{'label': 'Dark mode', 'value': 'dark'}],
            value=['first'],
            className='theme-switch'
        )
    ]),
    html.Div([
        html.Div([
            dcc.Link("Input", href='/input', className='nav-link'),
            dcc.Link("Exploration", href='/exploration', className='nav-link'),
            dcc.Link("Modelling", href='/modelling', className='nav-link'),
            dcc.Link("Results", href='/output', className='nav-link')
        ], className='sidebar', id='ContentsTable'),

        html.Div(dash.page_container, className='content')
    ], className='layout-container')
], id='app')

from dash.dependencies import Input, Output

@app.callback(
    Output('app', 'data-theme'),
    Output('theme', 'data'),
    Output('theme-toggle', 'value'),
    Input('theme-toggle', 'value'),
    Input('theme', 'data')
)
def update_theme(theme, global_theme):
    if 'first' in theme:
        pio.templates.default = "plotly_white"
        theme = global_theme.get('theme') if global_theme else []
        global_theme = {'theme': theme}
        return theme, global_theme, theme
    
    global_theme = {'theme': theme} if not global_theme else global_theme
    
    try:
        if 'dark' in theme:
            pio.templates.default = "plotly_dark"
            theme = ['dark']
            global_theme = {'theme': ['dark']}
        else:
            pio.templates.default = "plotly_white"
            theme = []
            global_theme = {'theme': []}
    except:
        pio.templates.default = "plotly_white"
        theme = []
        global_theme = {'theme': []}
    
    return theme, global_theme, theme

if __name__ == "__main__":
    app.run(debug=True)
