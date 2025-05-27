from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash

import base64
import datetime
import io

import pandas as pd

dash.register_page(__name__)

layout = html.Div([
    # dcc.Store(id='dataset', storage_type='local'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Don't allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload'),
])

@callback(Output('dataset', 'data'),
          Input('upload-data', 'contents'),
          Input('dataset', 'data'),
          State('upload-data', 'filename'),
          prevent_initial_call=False)
def update_dataset(contents, dataset, filename):
    if contents:
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return None
        return df.to_json(date_format='iso', orient='split')
    else:
        return dataset

@callback(Output('output-data-upload', 'children'),
          Input('dataset', 'data'),
          Input('theme-toggle', 'value'),
          prevent_initial_call=False)
def update_output(dataset_json, theme_value):
    if dataset_json is None:
        return html.Div("No data uploaded yet.")

    # ✅ Load the DataFrame from the JSON string
    df = pd.read_json(io.StringIO(dataset_json), orient='split')

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


    return html.Div([
        dash_table.DataTable(
            data=df.to_dict('records'),  # ✅ Pass list of dicts, not the JSON string
            columns=[{'name': i, 'id': i} for i in df.columns],
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
            ]
        )
    ])