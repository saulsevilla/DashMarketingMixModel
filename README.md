# DashMarketingMixModel

Esta es una aplicación interactiva construida con [Dash](https://dash.plotly.com/) para ajustar y visualizar un modelo de Marketing Mix utilizando técnicas bayesianas. Permite analizar la contribución de distintos canales de inversión, el ROI, la saturación de medios y más.

---

## 🚀 Características

- Selección dinámica de variables de ventas y medios.
- Ajuste de parámetros como `decay` y número de componentes de Fourier.
- Modelo bayesiano basado en adstock + curva Hill.
- Visualización de:
  - Contribución por canal.
  - ROI (Retorno sobre Inversión).
  - Saturación de medios.
  - Betas estimadas.
  - Ventas observadas vs esperadas.
- Soporte para temas claro y oscuro.

---

## 🛠 Requisitos

- Python 3.11+
- Recomendado: crear un entorno virtual

# Cómo ejecutar
Clona el repositorio:
```bash
git clone https://github.com/saulsevilla/DashMarketingMixModel.git
cd DashMarketingMixModel
```
Crear un entorno virtual (opcional):
```bash
python -m venv venv
```
Activa tu entorno virtual (si tienes uno):

```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

Instala las dependencias:
```bash
pip install -r requirements.txt
```

Ejecuta la aplicación:

```bash
python app.py
```

Abre tu navegador en: [http://127.0.0.1:8050]

