import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.io as pio
import plotly.graph_objs as go
from sklearn.linear_model import BayesianRidge

# === FUNCIONES ===

def apply_adstock(x, decay):
    result = np.zeros_like(x)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + decay * result[t - 1]
    return result

def generate_dummy_data():
    np.random.seed(123)
    weeks = pd.date_range("2022-01-01", periods=104, freq="W")
    data = pd.DataFrame({
        "date": weeks,
        "sales": 100 + np.cumsum(np.random.normal(0, 1, 104)) + np.random.normal(0, 10, 104),
        "tv": np.abs(np.random.normal(100, 20, 104)),
        "digital": np.abs(np.random.normal(80, 15, 104)),
        "print": np.abs(np.random.normal(50, 10, 104))
    })
    return data

def preprocess_data(df, decay_tv, decay_digital, decay_print, n_fourier):
    df = df.copy()
    df["tv_adstock"] = apply_adstock(df["tv"].values, decay_tv)
    df["digital_adstock"] = apply_adstock(df["digital"].values, decay_digital)
    df["print_adstock"] = apply_adstock(df["print"].values, decay_print)

    df["week"] = (df["date"] - df["date"].min()).dt.days // 7
    for k in range(1, n_fourier + 1):
        df[f"S{k}"] = np.sin(2 * np.pi * k * df["week"] / 52)
        df[f"C{k}"] = np.cos(2 * np.pi * k * df["week"] / 52)
    
    return df

def fit_mmm_model(X, y):
    """
    Ajusta un modelo Bayesian Ridge para Marketing Mix Modeling.
    
    Parameters:
    - X: DataFrame de predictores.
    - y: Series o array de la variable de ventas.
    
    Returns:
    - model: Objeto BayesianRidge entrenado.
    - fitted_values: Ventas predichas.
    - coefficients: Coeficientes (sin incluir el intercepto).
    """
    model = BayesianRidge()
    model.fit(X, y)

    fitted_values = model.predict(X)
    coefficients = pd.Series(model.coef_, index=X.columns)

    return model, fitted_values, coefficients

# === APP ===

dash.register_page(__name__)

layout = html.Div([
    html.H2("Marketing Mix Model (MMM) con Adstock y Estacionalidad"),

    html.Div([
        html.H4("Configuración del Modelo MMM"),

        dcc.Loading([
            html.Div([
                html.Label("Variable de Ventas"),
                dcc.Dropdown(id="sales-column", placeholder="Selecciona variable de ventas",
                             className="custom-dropdown")
            ], style={"marginBottom": "10px"}),

            html.Div([
                html.Label("Variables de Inversión (Medios)"),
                dcc.Dropdown(id="media-columns", multi=True, placeholder="Selecciona variables de inversión",
                             className="custom-dropdown")
            ], style={"marginBottom": "10px"}),

            html.Div([
                html.Label("Adstock Decay por Canal"),
                html.Div(id="adstock-decay-inputs")
            ], style={"marginBottom": "20px"}),

            html.Div([
                html.Label("Componentes de Fourier"),
                dcc.Input(id="n-fourier", type="number", min=0, max=100, step=1, value=0)
            ], style={"marginBottom": "20px"}),

            html.Button("Ajustar Modelo", id="run-model", n_clicks=0)
        ], type="default")
    ], style={"padding": "20px", "border": "1px solid #ccc", "borderRadius": "8px", "width": "350px"}),
    html.H1("Vista previa del modelo"),
    dcc.Graph(id="fitted-vs-actual"),

    # html.H4("Parámetros del Modelo"),
    # html.Pre(id="model-params", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"})
])

@callback(
    Output('n-fourier', 'value'),
    Input('n-fourier', 'value'),
    Input('model', 'data'),
)
def fourier_comps(n_fourier, model):
    print(n_fourier)
    print(model.get('fourier', 0))
    n_fourier = model.get('fourier', 0) if n_fourier == 0 and model else n_fourier
    print(n_fourier)
    return n_fourier

@callback(
    Output("adstock-decay-inputs", "children"),
    Input("media-columns", "value"),
    Input("model", "data")
)
def render_adstock_inputs(media_columns, model):
    adstock_map = model.get('adstock', dict()) if model else dict()

    if not media_columns:
        return html.Div("Selecciona al menos un canal.")

    return [
        html.Div([
            html.Label(f"Decay - {col}"),
            dcc.Input(
                id={"type": "decay-input", "index": col},
                type="number", min=0, max=1, step=0.05, value=adstock_map.get(col, 0.5)
            )
        ], style={"marginBottom": "10px"}) for col in media_columns
    ]

import io
@callback(
    [Output("sales-column", "options"),
     Output("sales-column", "value"),
     Output("media-columns", "options"),
     Output("media-columns", "value")],
    Input("dataset", "data"),
    Input("model", "data"),
    Input('theme-toggle', 'value')
)
def update_variable_options(data, model, theme):
    media_cols = model.get('predictor_columns', []) if model else []
    sales_col = model.get('sales_col', None) if model else None
    
    if not data:
        return [], []
    df = pd.read_json(io.StringIO(data), orient='split')
    columns = df.columns

    is_dark = 'dark' in theme
    if is_dark:
        bg = '#1f1f1f'
        text_color = '#e0e0e0'
    else:
        bg = '#ffffff'
        text_color = '#212529'
    
    style = {
        "backgroundColor": bg,
        "color": text_color,
        "border": "0px solid",
        "borderRadius": "0px",
        "width": "100%"
    }

    options = [{"label": html.Span(col, style= style), "value": col} for col in columns if col != 'Unnamed: 0']
    return options, sales_col, options, media_cols


from dash.dependencies import ALL

@callback(
    Output("fitted-vs-actual", "figure"),
    Output("model", "data"),
    Input('theme-toggle', 'value'),
    Input("run-model", "n_clicks"),
    State("sales-column", "value"),
    State("media-columns", "value"),
    State({"type": "decay-input", "index": ALL}, "value"),
    State({"type": "decay-input", "index": ALL}, "id"),
    State("n-fourier", "value"),
    State("dataset", "data")
)
def ajustar_modelo(theme_value, n_clicks, sales_col, media_cols, decay_values, decay_ids, n_fourier, dataset):
    if 'dark' in theme_value:
        pio.templates.default = "plotly_dark"
    else:
        pio.templates.default = "plotly_white"

    if not all([sales_col, media_cols, decay_values, dataset]):
        print([bool(sales_col), bool(media_cols), bool(decay_values), bool(dataset)])
        return "Faltan datos."

    # Mapeo: canal -> decay
    decay_map = {item["index"]: value for item, value in zip(decay_ids, decay_values)}

    # Dataset a DataFrame
    import pandas as pd
    df = pd.read_json(io.StringIO(dataset), orient='split')

    # Aplicar Adstock por canal
    def apply_adstock(x, decay):
        adstocked = [x[0]]
        for i in range(1, len(x)):
            adstocked.append(x[i] + decay * adstocked[i - 1])
        return adstocked

    for col in media_cols:
        df[f"{col}_adstock"] = apply_adstock(df[col].fillna(0).values, decay_map.get(col, 0.5))

    model, results, params = fit_mmm_model(df[media_cols], df[sales_col])

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df[sales_col], mode="lines", name="Ventas reales"))
    fig.add_trace(go.Scatter(y=results, mode="lines", name="Ventas predichas"))
    fig.update_layout(title="Ventas reales vs. predicción", xaxis_title="Fecha", yaxis_title="Ventas")

    # Parámetros
    # params_str = model.summary().as_text()

    # Modelo
    # print(model.params),
    model = {
        'betas': params,
        'sales_col': sales_col,
        'predictor_columns': media_cols,
        'adstock': decay_map,
        'fourier': n_fourier,
        'y_true': df[sales_col],
        'y_pred': results
    }
    
    return fig, model

    return f"Modelo ajustado con {len(media_cols)} canales y {n_fourier} componentes Fourier."

@callback(
    Output("sales-column", "style"),
    Output("media-columns", "style"),
    Output("n-fourier", "style"),
    Output("run-model", "style"),
    Input('theme-toggle', 'value')
)
def styles(theme):
    # Check if dark mode is enabled
    is_dark = 'dark' in theme

    # Choose colors based on theme
    if is_dark:
        bg = '#1f1f1f'
        text_color = '#e0e0e0'
    else:
        bg = '#ffffff'
        text_color = '#212529'
    
    style = {
        "backgroundColor": bg,
        "color": text_color,
        "border": "1px solid #444",
        "borderRadius": "5px",
    }
    
    return style, style, style, style