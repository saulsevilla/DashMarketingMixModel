import dash
from dash import html

dash.register_page(__name__, path='/')

layout = html.Div([
    html.H1('Proyecto Final'),
    html.Div('Sevilla Gallardo Saúl Sebastián'),
    html.Div('Marketing - Fernando Soto'),
], className='content')