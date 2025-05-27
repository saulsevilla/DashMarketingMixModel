import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx, callback, dash_table
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np

import io

dash.register_page(__name__)

layout = html.Div([html.Br(),
    html.H1("Contribución por canal"),
    dcc.Graph(id='channel-contribution'),

    html.H1("ROI por canal"),
    dcc.Graph(id='channel-roi'),

    html.H1("Saturación por canal"),
    dcc.Graph(id='saturation-plot'),

    html.H1("Betas del Modelo"),
    html.Div(id='model-betas'),

    html.H1("Ventas: Observadas vs Esperadas"),
    dcc.Graph(id='observed-vs-predicted'),
])

@callback(
    Output('channel-contribution', 'figure'),
    Output('channel-roi', 'figure'),
    Output('saturation-plot', 'figure'),
    Output('model-betas', 'children'),
    Output('observed-vs-predicted', 'figure'),
    Input('model', 'data'),
    Input('dataset', 'data'),
    Input('theme-toggle', 'value')
)
def update_model_results(model_data, dataset, theme_value):
    
    if 'dark' in theme_value:
        pio.templates.default = "plotly_dark"
    else:
        pio.templates.default = "plotly_white"
    if not model_data or not dataset:
        raise PreventUpdate

    df = pd.read_json(io.StringIO(dataset), orient='split')
    betas = pd.Series(model_data['betas'], index=model_data['predictor_columns'])
    y_true = pd.Series(model_data['y_true'])
    y_pred = pd.Series(model_data['y_pred'])
    
    # Contribución
    X = df[model_data['predictor_columns']]
    contribution = X.mul(betas, axis=1)
    total_contribution = contribution.sum()
    fig_contribution = px.bar(total_contribution, labels={'index': 'Canal', 'value': 'Contribución total'})

    # ROI = contribución / gasto
    gastos = X.sum()
    roi = total_contribution / gastos
    fig_roi = px.bar(roi, labels={'index': 'Canal', 'value': 'ROI'})

    # Saturación: inversión vs respuesta marginal
    # Parámetros de saturación (puedes hacer estos dinámicos si lo deseas)
    hill_params = {
        'n': 1.5,      # parámetro de forma
        'theta': 0.5   # inversión en la cual se alcanza 50% de respuesta
    }

    fig_saturation = go.Figure()

    for idx, col in enumerate(model_data['predictor_columns']):
        x_vals = X[col].values
        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)

        beta = model_data['betas'][idx]
        n = hill_params['n']
        theta = hill_params['theta'] * x_vals.max()  # escala relativa

        # Aplicar función de Hill
        y_vals = beta * (x_range ** n) / (x_range ** n + theta ** n)

        fig_saturation.add_trace(go.Scatter(
            x=x_range,
            y=y_vals,
            mode='lines',
            name=col
        ))

    fig_saturation.update_layout(
        title='Curva de Saturación (Hill)',
        xaxis_title='Inversión por canal',
        yaxis_title='Respuesta esperada',
    )

    # Check if dark mode is enabled
    is_dark = 'dark' in theme_value

    # Choose colors based on theme
    if is_dark:
        cell_bg = '#1f1f1f'
        text_color = '#e0e0e0'
        header_bg = '#121212'
        header_color = '#ffffff'
    else:
        cell_bg = '#ffffff'
        text_color = '#212529'
        header_bg = '#e9ecef'
        header_color = '#000000'

    # Betas
    beta_table = dash_table.DataTable(
        data = [{'Canal': canal, 'Beta': beta} for canal, beta in betas.items()],
        columns= [{'name': 'Canal', 'id': 'Canal'}, {'name': 'Beta', 'id': 'Beta'}],
        style_table={
                'overflowX': 'auto',
                'maxWidth': '100%',
            },
            style_cell={
                'maxWidth': '200px',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'backgroundColor': cell_bg,
                'color': text_color,
            },
            style_header={
                'backgroundColor': header_bg,
                'color': header_color,
                'fontWeight': 'bold',
            },
            style_data_conditional=[
                {
                    'if': {'state': 'active'},  # Hovered row
                    'backgroundColor': '#2c2c2c' if is_dark else '#f1f1f1',
                    'color': '#ffffff' if is_dark else '#000000'
                },
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#1a1a1a' if is_dark else '#f9f9f9'
                }
            ],
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in df.to_dict('records')
            ],
            tooltip_delay = 1,
            tooltip_duration=None,
            css=[
                {
                    'selector': '.dash-table-tooltip',
                    'rule': (
                        'background-color: #333333; color: #ffffff; font-family: monospace;'
                        if is_dark else
                        'background-color: #f9f9f9; color: #000000; font-family: monospace;'
                    )
                }
            ])
    
    

    # Observado vs Predicho
    # Gráfico
    fig_obs_pred = go.Figure()
    fig_obs_pred.add_trace(go.Scatter(y=y_true, mode="lines", name="Ventas reales"))
    fig_obs_pred.add_trace(go.Scatter(y=y_pred, mode="lines", name="Ventas predichas"))

    # fig.update_layout(title="Ventas reales vs. predicción", xaxis_title="Fecha", yaxis_title="Ventas")
    # fig_obs_pred = px.line(x=list(range(len(y_true))), y=y_true, labels={'x': 'Tiempo', 'y': 'Ventas'})
    # fig_obs_pred.add_scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='Predicción')

    return fig_contribution, fig_roi, fig_saturation, beta_table, fig_obs_pred