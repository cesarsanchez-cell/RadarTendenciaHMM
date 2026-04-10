"""
Radar de Tendencia HMM
======================
Monitor de mercado con detección de regímenes usando Hidden Markov Models.
Aplicación Streamlit con GaussianHMM multivariado, candlestick interactivo
estilo TradingView y matriz de transición.

Ejecutar: streamlit run app.py
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM

# ============================================================================
# SECTION A: Configuration Constants
# ============================================================================

TIMEFRAME_CONFIG = {
    "1H": {
        "yf_interval": "1h",
        "yf_period": "730d",
        "resample_rule": None,
        "vol_window": 20,
        "momentum_ema_span": 14,
        "momentum_slope_window": 5,
        "min_bars_required": 100,
        "label": "1 Hora",
    },
    "4H": {
        "yf_interval": "1h",
        "yf_period": "730d",
        "resample_rule": "4h",
        "vol_window": 20,
        "momentum_ema_span": 14,
        "momentum_slope_window": 5,
        "min_bars_required": 100,
        "label": "4 Horas",
    },
    "1D": {
        "yf_interval": "1d",
        "yf_period": "5y",
        "resample_rule": None,
        "vol_window": 20,
        "momentum_ema_span": 14,
        "momentum_slope_window": 5,
        "min_bars_required": 100,
        "label": "1 Dia",
    },
    "1W": {
        "yf_interval": "1d",
        "yf_period": "10y",
        "resample_rule": "1W",
        "vol_window": 12,
        "momentum_ema_span": 10,
        "momentum_slope_window": 4,
        "min_bars_required": 60,
        "label": "1 Semana",
    },
}

# Paleta de colores por regimen: de bear (rojo) a bull (verde)
REGIME_PALETTES = {
    3: ["#F44336", "#78909C", "#4CAF50"],
    4: ["#E53935", "#EF5350", "#66BB6A", "#43A047"],
    5: ["#E53935", "#EF5350", "#78909C", "#66BB6A", "#43A047"],
    6: ["#C62828", "#E53935", "#EF5350", "#66BB6A", "#43A047", "#2E7D32"],
    7: ["#C62828", "#E53935", "#EF5350", "#78909C", "#66BB6A", "#43A047", "#2E7D32"],
}

# Labels semanticos por cantidad de regimenes
REGIME_LABELS = {
    3: ["Bear", "Sideways", "Bull"],
    4: ["Strong Bear", "Bear", "Bull", "Strong Bull"],
    5: ["Strong Bear", "Bear", "Sideways", "Bull", "Strong Bull"],
    6: ["Strong Bear", "Bear", "Mild Bear", "Mild Bull", "Bull", "Strong Bull"],
    7: ["Strong Bear", "Bear", "Mild Bear", "Sideways", "Mild Bull", "Bull", "Strong Bull"],
}

# Emojis por label
REGIME_EMOJIS = {
    "Strong Bear": "🐻",
    "Bear": "🐻",
    "Mild Bear": "📉",
    "Sideways": "➡️",
    "Mild Bull": "📈",
    "Bull": "🐂",
    "Strong Bull": "🐂",
}

MATH_EXPLANATION = r"""
### Enfoque Matematico

**Hidden Markov Model (HMM)** — El mercado se modela como un sistema con *K* estados
ocultos (no observables), donde cada estado representa un **regimen de mercado** distinto.
En cada paso temporal, el sistema ocupa un estado y emite un vector de observacion
extraido de una distribucion Gaussiana multivariada especifica de ese estado.

**Formalmente:**
- Estados ocultos: $S = \{s_1, ..., s_K\}$ (los regimenes)
- Matriz de transicion: $A$ donde $A_{ij} = P(state_t = s_j \mid state_{t-1} = s_i)$
- Distribuciones de emision: cada estado $s_k$ tiene parametros $(\mu_k, \Sigma_k)$

**El vector de observacion** tiene 3 dimensiones:

$$\mathbf{x}_t = [r_t, \sigma_t, m_t]$$

1. **Retorno logaritmico** $r_t = \ln(C_t / C_{t-1})$ — Captura la direccion del precio.
   Son aditivos en el tiempo y aproximadamente normales para valores pequenos, lo que
   se alinea con la suposicion Gaussiana del HMM.

2. **Volatilidad realizada** $\sigma_t = \text{std}(r_{t-w+1}, ..., r_t)$ — Desviacion estandar
   movil de los retornos sobre una ventana $w$. Separa regimenes de alta/baja volatilidad
   (crisis vs. tendencias estables).

3. **Momentum direccional** — Pendiente de la EMA:
   $m_t = \frac{EMA_t - EMA_{t-k}}{k}$, donde $EMA_t = \alpha \cdot C_t + (1-\alpha) \cdot EMA_{t-1}$
   y $\alpha = 2/(span+1)$. Es continuo e ilimitado, ideal para emisiones Gaussianas.

**Entrenamiento:** Algoritmo Baum-Welch (Expectation-Maximization) para estimar $A$, $\mu_k$ y $\Sigma_k$.

**Decodificacion:** Algoritmo de Viterbi para la secuencia mas probable de estados ocultos.

**Normalizacion:** Se usa `RobustScaler` (mediana + IQR) en vez de `StandardScaler`
para ser robusto a los fat tails tipicos de datos financieros.

**Covarianza completa** (`covariance_type="full"`) captura correlaciones entre features,
como el *leverage effect* (retornos negativos - volatilidad creciente).
"""

# ============================================================================
# SECTION A.1: TradingView-style CSS Theme
# ============================================================================

TRADINGVIEW_CSS = """
<style>
    /* ── TradingView Dark Theme ── */
    .stApp {
        background-color: #131722;
        color: #D1D4DC;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1E222D;
        border-right: 1px solid #2A2E39;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #D1D4DC;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #D1D4DC !important;
        font-family: 'Trebuchet MS', 'Segoe UI', sans-serif !important;
    }
    h1 {
        font-size: 2rem !important;
        letter-spacing: -0.5px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Consolas', 'Monaco', monospace !important;
        font-size: 1.3rem !important;
        color: #D1D4DC !important;
    }
    [data-testid="stMetricLabel"] {
        color: #787B86 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Cards */
    .regime-card {
        background: linear-gradient(135deg, #1E222D 0%, #131722 100%);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
        border: 1px solid #2A2E39;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .regime-card h2 {
        margin: 0 0 4px 0;
        font-size: 2.2rem !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .regime-card .subtitle {
        color: #787B86;
        font-size: 1rem;
        margin: 0;
    }
    .regime-card .metrics-row {
        display: flex;
        gap: 32px;
        margin-top: 16px;
        flex-wrap: wrap;
    }
    .regime-card .metric-item {
        display: flex;
        flex-direction: column;
    }
    .regime-card .metric-label {
        color: #787B86;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 2px;
    }
    .regime-card .metric-value {
        color: #D1D4DC;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Transition probability bars */
    .prob-bar-container {
        margin-bottom: 12px;
        padding: 8px 12px;
        background: #1E222D;
        border-radius: 8px;
        border: 1px solid #2A2E39;
    }
    .prob-bar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .prob-bar-label {
        font-weight: 600;
        font-size: 0.95rem;
    }
    .prob-bar-value {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .prob-bar-track {
        height: 8px;
        background: #2A2E39;
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Distribution badges */
    .dist-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #2A2E39;
        background: #1E222D;
    }
    .dist-badge .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }

    /* Section titles */
    .section-title {
        color: #787B86;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2A2E39;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dividers */
    hr {
        border-color: #2A2E39 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #131722;
    }
    ::-webkit-scrollbar-thumb {
        background: #2A2E39;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #787B86;
    }

    /* Responsive: stack columns on small screens */
    @media (max-width: 768px) {
        .regime-card h2 {
            font-size: 1.6rem !important;
        }
        .regime-card .metrics-row {
            gap: 16px;
        }
        .regime-card .metric-value {
            font-size: 1rem;
        }
    }

    /* Button styling */
    .stButton > button {
        background-color: #2962FF !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #1E88E5 !important;
        box-shadow: 0 4px 12px rgba(41,98,255,0.3) !important;
    }

    /* Selectbox, slider, text input */
    .stSelectbox label, .stSlider label, .stTextInput label {
        color: #787B86 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
"""


# ============================================================================
# SECTION B: Data Ingestion
# ============================================================================

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resamplea datos OHLCV a un timeframe superior."""
    agg_dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    resampled = df.resample(rule).agg(agg_dict)
    resampled.dropna(subset=["Close"], inplace=True)
    return resampled


def fetch_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """
    Obtiene datos OHLCV. Detecta si es crypto (contiene '/') para usar ccxt,
    sino usa yfinance. Aplica resample si el timeframe lo requiere.
    """
    config = TIMEFRAME_CONFIG[timeframe]
    is_crypto = "/" in ticker

    if is_crypto:
        df = _fetch_crypto(ticker, timeframe, config)
    else:
        df = _fetch_traditional(ticker, config)

    if config["resample_rule"] is not None:
        df = resample_ohlcv(df, config["resample_rule"])

    if len(df) < config["min_bars_required"]:
        raise ValueError(
            f"Solo {len(df)} barras disponibles. Se requieren al menos "
            f"{config['min_bars_required']} para el timeframe {timeframe}. "
            f"Intenta con un timeframe mas largo o un ticker con mas historia."
        )

    return df


def _fetch_traditional(ticker: str, config: dict) -> pd.DataFrame:
    """Obtiene datos de acciones/indices via yfinance."""
    import yfinance as yf

    data = yf.download(
        ticker,
        period=config["yf_period"],
        interval=config["yf_interval"],
        progress=False,
        auto_adjust=True,
    )

    if data is None or data.empty:
        raise ValueError(
            f"No se pudieron obtener datos para '{ticker}'. "
            f"Verifica que el simbolo sea correcto (ej: SPY, AAPL, BTC-USD)."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    col_map = {}
    for col in data.columns:
        col_lower = str(col).lower()
        if "open" in col_lower:
            col_map[col] = "Open"
        elif "high" in col_lower:
            col_map[col] = "High"
        elif "low" in col_lower:
            col_map[col] = "Low"
        elif "close" in col_lower:
            col_map[col] = "Close"
        elif "volume" in col_lower:
            col_map[col] = "Volume"
    if col_map:
        data = data.rename(columns=col_map)

    data.dropna(subset=["Close"], inplace=True)
    return data


def _fetch_crypto(ticker: str, timeframe: str, config: dict) -> pd.DataFrame:
    """Obtiene datos de crypto via ccxt (Binance)."""
    import ccxt

    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        tf_map = {"1H": "1h", "4H": "1h", "1D": "1d", "1W": "1d"}
        ccxt_tf = tf_map.get(timeframe, "1d")
        limit = 1000
        all_ohlcv = exchange.fetch_ohlcv(ticker, timeframe=ccxt_tf, limit=limit)

        if not all_ohlcv:
            raise ValueError(f"No se obtuvieron datos para '{ticker}' en Binance.")

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.index.name = None
        return df

    except Exception as e:
        if "ccxt" in str(type(e).__module__):
            raise ValueError(
                f"Error al obtener datos crypto para '{ticker}'. "
                f"Verifica el formato del par (ej: BTC/USDT). Error: {e}"
            )
        raise


# ============================================================================
# SECTION C: Feature Engineering
# ============================================================================

def compute_log_returns(close: pd.Series) -> pd.Series:
    """Calcula retornos logaritmicos: r_t = ln(C_t / C_{t-1})"""
    return np.log(close / close.shift(1))


def compute_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """Volatilidad realizada como desviacion estandar movil de retornos."""
    return log_returns.rolling(window=window).std()


def compute_momentum(close: pd.Series, ema_span: int, slope_window: int) -> pd.Series:
    """
    Momentum direccional como pendiente de la EMA.
    slope_t = (EMA_t - EMA_{t-k}) / k
    """
    ema = close.ewm(span=ema_span, adjust=False).mean()
    slope = (ema - ema.shift(slope_window)) / slope_window
    return slope


def compute_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calcula las 3 features para el HMM y elimina filas con NaN.
    Returns DataFrame con columnas: log_return, volatility, momentum
    """
    close = df["Close"].squeeze()
    log_ret = compute_log_returns(close)
    vol = compute_volatility(log_ret, window=config["vol_window"])
    mom = compute_momentum(
        close,
        ema_span=config["momentum_ema_span"],
        slope_window=config["momentum_slope_window"],
    )

    features = pd.DataFrame(
        {"log_return": log_ret, "volatility": vol, "momentum": mom},
        index=df.index,
    )
    features.dropna(inplace=True)

    if features.empty:
        raise ValueError(
            "No quedan datos despues de calcular features. "
            "Los datos son insuficientes para las ventanas moviles configuradas."
        )

    return features


def normalize_features(features: pd.DataFrame):
    """
    Normaliza features con RobustScaler (robusto a outliers / fat tails).
    Returns: (features_scaled ndarray, scaler fitted)
    """
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features.values)
    return scaled, scaler


# ============================================================================
# SECTION D: HMM Training Pipeline
# ============================================================================

def train_hmm(features_scaled: np.ndarray, n_regimes: int):
    """
    Entrena un GaussianHMM multivariado.
    Returns: (model, converged: bool)
    """
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        tol=0.01,
    )

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features_scaled)
        for w in caught:
            if "ConvergenceWarning" in str(w.category.__name__):
                converged = False

    return model, converged


def decode_regimes(model: GaussianHMM, features_scaled: np.ndarray) -> np.ndarray:
    """Decodifica la secuencia de regimenes usando el algoritmo de Viterbi."""
    return model.predict(features_scaled)


def label_regimes(model: GaussianHMM, n_regimes: int):
    """
    Ordena regimenes por media de log-return (feature 0) y asigna labels y colores.
    Returns: (label_map dict, color_map dict, sorted_order list)
    """
    mean_returns = model.means_[:, 0]
    sorted_indices = np.argsort(mean_returns)

    labels = REGIME_LABELS[n_regimes]
    colors = REGIME_PALETTES[n_regimes]

    label_map = {}
    color_map = {}
    for rank, original_idx in enumerate(sorted_indices):
        label_map[original_idx] = labels[rank]
        color_map[original_idx] = colors[rank]

    return label_map, color_map, sorted_indices


def get_transition_matrix(model: GaussianHMM, label_map: dict, sorted_order) -> pd.DataFrame:
    """Extrae la matriz de transicion como DataFrame con labels semanticos."""
    transmat = model.transmat_
    reordered = transmat[sorted_order][:, sorted_order]
    labels = [label_map[idx] for idx in sorted_order]
    return pd.DataFrame(reordered, index=labels, columns=labels)


def project_regimes(model: GaussianHMM, current_regime: int, n_periods: int,
                    label_map: dict, color_map: dict, sorted_order) -> pd.DataFrame:
    """
    Proyecta probabilidades de regimen hacia adelante N periodos
    usando potencias sucesivas de la matriz de transicion.

    En t+1: P(state) = transmat[current_regime]
    En t+k: P(state) = P(t+k-1) @ transmat  (propagacion iterativa)

    Returns DataFrame con columnas = regimenes, filas = periodos futuros (t+1..t+N)
    """
    transmat = model.transmat_
    n_states = model.n_components

    # Vector de estado inicial: 100% en el regimen actual
    state_vec = np.zeros(n_states)
    state_vec[current_regime] = 1.0

    projections = []
    for step in range(1, n_periods + 1):
        state_vec = state_vec @ transmat
        projections.append(state_vec.copy())

    # Crear DataFrame con labels semanticos, reordenado
    labels = [label_map[idx] for idx in sorted_order]
    proj_array = np.array(projections)
    # Reordenar columnas segun sorted_order
    proj_reordered = proj_array[:, sorted_order]

    df_proj = pd.DataFrame(
        proj_reordered,
        index=[f"t+{i}" for i in range(1, n_periods + 1)],
        columns=labels,
    )
    return df_proj


def _hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    """Convierte color hex (#RRGGBB) a rgba() string para Plotly."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_projection(proj_df: pd.DataFrame, color_map: dict, label_map: dict,
                    sorted_order, timeframe: str) -> go.Figure:
    """
    Grafico de area apilada mostrando la evolucion de probabilidades
    de regimen proyectadas hacia adelante.
    """
    fig = go.Figure()

    # Mapeo de timeframe a unidad legible
    tf_units = {"1H": "horas", "4H": "periodos de 4h", "1D": "dias", "1W": "semanas"}
    unit = tf_units.get(timeframe, "periodos")

    labels = proj_df.columns.tolist()
    colors = [color_map[idx] for idx in sorted_order]

    for i, (label, color) in enumerate(zip(labels, colors)):
        fig.add_trace(
            go.Scatter(
                x=proj_df.index,
                y=proj_df[label],
                name=label,
                mode="lines",
                line=dict(width=0.5, color=color),
                stackgroup="one",
                fillcolor=_hex_to_rgba(color, 0.4),
                hovertemplate=f"{label}: " + "%{y:.1%}<extra></extra>",
            )
        )

    fig.update_layout(
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#D1D4DC", size=13),
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis=dict(
            title=f"Periodos hacia adelante ({unit})",
            gridcolor="#1E222D",
            color="#787B86",
        ),
        yaxis=dict(
            title="Probabilidad",
            gridcolor="#1E222D",
            color="#787B86",
            tickformat=".0%",
            range=[0, 1],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#D1D4DC"),
            bgcolor="rgba(19,23,34,0.8)",
            bordercolor="#2A2E39",
            borderwidth=1,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1E222D", font_color="#D1D4DC", bordercolor="#2A2E39"),
    )

    return fig


def render_projection_summary(proj_df: pd.DataFrame, color_map: dict, label_map: dict,
                              sorted_order, n_periods: int, timeframe: str):
    """Renderiza un resumen de la proyeccion: regimen mas probable en t+1, t+mid, t+N."""
    tf_units = {"1H": "h", "4H": "x4h", "1D": "d", "1W": "w"}
    unit = tf_units.get(timeframe, "p")

    checkpoints = [0]  # t+1 siempre
    if n_periods > 2:
        checkpoints.append(n_periods // 2 - 1)  # mitad
    if n_periods > 1:
        checkpoints.append(n_periods - 1)  # final

    colors = [color_map[idx] for idx in sorted_order]
    labels = proj_df.columns.tolist()

    cards_html = '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
    for cp in checkpoints:
        row = proj_df.iloc[cp]
        top_regime = row.idxmax()
        top_prob = row.max()
        # Find color for top regime
        top_color = "#D1D4DC"
        for idx in sorted_order:
            if label_map[idx] == top_regime:
                top_color = color_map[idx]
                break
        emoji = REGIME_EMOJIS.get(top_regime, "")
        period_label = f"t+{cp + 1}"

        cards_html += (
            f'<div style="flex:1;min-width:140px;background:#1E222D;border-radius:8px;'
            f'padding:14px;border:1px solid #2A2E39;border-top:3px solid {top_color};">'
            f'<p style="color:#787B86;font-size:0.75rem;text-transform:uppercase;'
            f'letter-spacing:0.8px;margin:0 0 6px 0;">{period_label} ({(cp+1)}{unit})</p>'
            f'<p style="color:{top_color};font-size:1.1rem;font-weight:700;margin:0 0 2px 0;">'
            f'{emoji} {top_regime}</p>'
            f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
            f'font-size:1.3rem;font-weight:700;margin:0;">{top_prob:.1%}</p>'
            f'</div>'
        )
    cards_html += '</div>'
    st.html(cards_html)


# ============================================================================
# SECTION D.2: Grid/LP Range Projection (Square Root of Time)
# ============================================================================

def compute_range_projection(model: GaussianHMM, current_regime: int,
                             last_close: float, horizon: int,
                             z_score: float, scaler) -> dict:
    """
    Proyecta el rango operativo optimo para Grid/LP usando los parametros
    del regimen actual del HMM.

    Matematica:
    - mu_regime: media de log-returns del regimen actual (feature 0, escala original)
    - sigma_regime: desviacion estandar de log-returns del regimen actual
    - Escalamiento temporal: sigma_H = sigma_1 * sqrt(H) donde H = horizonte en periodos
    - Upper = last_close * exp(mu_H + z * sigma_H)
    - Lower = last_close * exp(mu_H - z * sigma_H)

    Los parametros del modelo estan en escala normalizada (RobustScaler),
    asi que necesitamos invertir la transformacion para obtener valores reales.
    """
    # Extraer media y covarianza del regimen actual en escala normalizada
    mean_scaled = model.means_[current_regime]      # shape (3,)
    covar_scaled = model.covars_[current_regime]     # shape (3, 3)

    # Invertir escalado para obtener valores en escala original
    # RobustScaler: x_scaled = (x - median) / IQR => x = x_scaled * IQR + median
    # scaler.center_ = medians, scaler.scale_ = IQR
    mu_original = mean_scaled[0] * scaler.scale_[0] + scaler.center_[0]
    # La varianza escalada se multiplica por IQR^2 para revertir
    sigma_original = np.sqrt(covar_scaled[0, 0]) * scaler.scale_[0]

    # Escalamiento por raiz cuadrada del tiempo
    mu_H = mu_original * horizon
    sigma_H = sigma_original * np.sqrt(horizon)

    # Bandas de confianza en espacio log
    upper_log = mu_H + z_score * sigma_H
    lower_log = mu_H - z_score * sigma_H

    # Convertir a precios
    upper_price = last_close * np.exp(upper_log)
    lower_price = last_close * np.exp(lower_log)
    mid_price = last_close * np.exp(mu_H)

    # Amplitud del rango en %
    amplitude_pct = (upper_price - lower_price) / last_close * 100

    # Generar puntos intermedios para el tunel visual (cada periodo)
    tunnel_upper = []
    tunnel_lower = []
    tunnel_mid = []
    for h in range(1, horizon + 1):
        mu_h = mu_original * h
        sigma_h = sigma_original * np.sqrt(h)
        tunnel_upper.append(last_close * np.exp(mu_h + z_score * sigma_h))
        tunnel_lower.append(last_close * np.exp(mu_h - z_score * sigma_h))
        tunnel_mid.append(last_close * np.exp(mu_h))

    return {
        "upper": upper_price,
        "lower": lower_price,
        "mid": mid_price,
        "amplitude_pct": amplitude_pct,
        "mu_original": mu_original,
        "sigma_original": sigma_original,
        "sigma_H": sigma_H,
        "z_score": z_score,
        "horizon": horizon,
        "last_close": last_close,
        "tunnel_upper": tunnel_upper,
        "tunnel_lower": tunnel_lower,
        "tunnel_mid": tunnel_mid,
    }


def add_range_tunnel_to_chart(fig: go.Figure, range_data: dict,
                               last_date, timeframe: str):
    """
    Agrega el tunel de rango proyectado al grafico de velas.
    Extiende hacia la derecha (futuro) con area sombreada + lineas punteadas.
    """
    horizon = range_data["horizon"]
    tunnel_upper = range_data["tunnel_upper"]
    tunnel_lower = range_data["tunnel_lower"]
    tunnel_mid = range_data["tunnel_mid"]
    last_close = range_data["last_close"]

    # Generar fechas futuras
    tf_deltas = {
        "1H": pd.Timedelta(hours=1),
        "4H": pd.Timedelta(hours=4),
        "1D": pd.Timedelta(days=1),
        "1W": pd.Timedelta(weeks=1),
    }
    delta = tf_deltas.get(timeframe, pd.Timedelta(days=1))
    future_dates = [last_date + delta * i for i in range(1, horizon + 1)]

    # Punto de inicio = ultimo cierre
    all_dates = [last_date] + future_dates
    all_upper = [last_close] + tunnel_upper
    all_lower = [last_close] + tunnel_lower
    all_mid = [last_close] + tunnel_mid

    # Area sombreada (tunel)
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=all_upper,
            mode="lines",
            line=dict(color="#FFD54F", width=2, dash="dash"),
            name=f"Upper ({range_data['z_score']}σ)",
            showlegend=True,
            hovertemplate="Upper: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=all_lower,
            mode="lines",
            line=dict(color="#FFD54F", width=2, dash="dash"),
            name=f"Lower ({range_data['z_score']}σ)",
            fill="tonexty",
            fillcolor="rgba(255, 213, 79, 0.08)",
            showlegend=True,
            hovertemplate="Lower: $%{y:,.2f}<extra></extra>",
        )
    )

    # Linea media esperada
    fig.add_trace(
        go.Scatter(
            x=all_dates,
            y=all_mid,
            mode="lines",
            line=dict(color="#FFD54F", width=1, dash="dot"),
            name="Expected",
            showlegend=False,
            hovertemplate="Expected: $%{y:,.2f}<extra></extra>",
        )
    )

    # Anotaciones de precio en el extremo derecho
    last_future = future_dates[-1]
    fig.add_annotation(
        x=last_future, y=tunnel_upper[-1],
        text=f"<b>${tunnel_upper[-1]:,.2f}</b>",
        showarrow=False,
        font=dict(size=11, color="#FFD54F"),
        bgcolor="#131722",
        xanchor="left", xshift=5,
    )
    fig.add_annotation(
        x=last_future, y=tunnel_lower[-1],
        text=f"<b>${tunnel_lower[-1]:,.2f}</b>",
        showarrow=False,
        font=dict(size=11, color="#FFD54F"),
        bgcolor="#131722",
        xanchor="left", xshift=5,
    )

    return fig


def render_range_card(range_data: dict, current_label: str, current_color: str, timeframe: str):
    """Renderiza la tarjeta de rango proyectado para Grid/LP."""
    tf_units = {"1H": "hora(s)", "4H": "periodo(s) 4H", "1D": "dia(s)", "1W": "semana(s)"}
    unit = tf_units.get(timeframe, "periodo(s)")

    upper = range_data["upper"]
    lower = range_data["lower"]
    mid = range_data["mid"]
    amp = range_data["amplitude_pct"]
    horizon = range_data["horizon"]
    z = range_data["z_score"]
    last = range_data["last_close"]

    # Color del rango segun amplitud (verde=estrecho/bueno, amarillo=medio, rojo=amplio)
    if amp < 5:
        amp_color = "#4CAF50"
        amp_label = "Estrecho"
    elif amp < 15:
        amp_color = "#FFD54F"
        amp_label = "Moderado"
    else:
        amp_color = "#EF5350"
        amp_label = "Amplio"

    st.html(
        f'<div style="background:linear-gradient(135deg,#1E222D 0%,#131722 100%);'
        f'border-radius:12px;padding:24px 28px;margin-bottom:16px;'
        f'border:1px solid #2A2E39;border-top:3px solid #FFD54F;'
        f'box-shadow:0 4px 24px rgba(0,0,0,0.3);">'
        # Titulo
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">'
        f'<div>'
        f'<p style="color:#FFD54F;font-size:0.75rem;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 4px 0;">📐 Rango Proyectado Grid/LP</p>'
        f'<p style="color:#787B86;font-size:0.8rem;margin:0;">'
        f'Horizonte: {horizon} {unit} | Z-score: {z}σ | Regimen: {current_label}</p>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<p style="color:{amp_color};font-size:1.8rem;font-weight:700;'
        f'font-family:Consolas,Monaco,monospace;margin:0;">{amp:.1f}%</p>'
        f'<p style="color:#787B86;font-size:0.75rem;margin:0;">Amplitud ({amp_label})</p>'
        f'</div></div>'
        # Grid de precios
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
        # Upper
        f'<div style="flex:1;min-width:120px;background:#131722;border-radius:8px;'
        f'padding:12px 16px;border:1px solid #2A2E39;">'
        f'<p style="color:#EF5350;font-size:0.7rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 4px 0;">▲ Upper Bound</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.3rem;font-weight:700;margin:0;">${upper:,.2f}</p>'
        f'<p style="color:#EF5350;font-size:0.8rem;margin:2px 0 0 0;">'
        f'+{((upper/last)-1)*100:.2f}%</p></div>'
        # Mid / Expected
        f'<div style="flex:1;min-width:120px;background:#131722;border-radius:8px;'
        f'padding:12px 16px;border:1px solid #2A2E39;">'
        f'<p style="color:#787B86;font-size:0.7rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 4px 0;">◆ Expected</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.3rem;font-weight:700;margin:0;">${mid:,.2f}</p>'
        f'<p style="color:#787B86;font-size:0.8rem;margin:2px 0 0 0;">'
        f'{((mid/last)-1)*100:+.2f}%</p></div>'
        # Lower
        f'<div style="flex:1;min-width:120px;background:#131722;border-radius:8px;'
        f'padding:12px 16px;border:1px solid #2A2E39;">'
        f'<p style="color:#4CAF50;font-size:0.7rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 4px 0;">▼ Lower Bound</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.3rem;font-weight:700;margin:0;">${lower:,.2f}</p>'
        f'<p style="color:#4CAF50;font-size:0.8rem;margin:2px 0 0 0;">'
        f'{((lower/last)-1)*100:.2f}%</p></div>'
        # Last close
        f'<div style="flex:1;min-width:120px;background:#131722;border-radius:8px;'
        f'padding:12px 16px;border:1px solid #2A2E39;">'
        f'<p style="color:#787B86;font-size:0.7rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 4px 0;">● Precio Actual</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.3rem;font-weight:700;margin:0;">${last:,.2f}</p>'
        f'<p style="color:#787B86;font-size:0.8rem;margin:2px 0 0 0;">'
        f'σ diaria: {range_data["sigma_original"]:.4f}</p></div>'
        f'</div></div>'
    )


# ============================================================================
# SECTION E: Visualization — TradingView Style
# ============================================================================

def plot_candlestick_with_regimes(
    df: pd.DataFrame,
    regimes: np.ndarray,
    label_map: dict,
    color_map: dict,
    feature_index: pd.Index,
) -> go.Figure:
    """
    Grafico de velas estilo TradingView con fondo coloreado por regimen.
    Crosshair, grid sutil, colores pro, regimenes con labels visibles.
    """
    df_aligned = df.loc[feature_index]

    fig = go.Figure()

    # ── Candlestick con estilo TradingView ──
    fig.add_trace(
        go.Candlestick(
            x=df_aligned.index,
            open=df_aligned["Open"].squeeze(),
            high=df_aligned["High"].squeeze(),
            low=df_aligned["Low"].squeeze(),
            close=df_aligned["Close"].squeeze(),
            name="OHLC",
            increasing_line_color="#26A69A",
            increasing_fillcolor="#26A69A",
            decreasing_line_color="#EF5350",
            decreasing_fillcolor="#EF5350",
            increasing_line_width=1.5,
            decreasing_line_width=1.5,
            whiskerwidth=0.4,
        )
    )

    # ── Regimenes: vrects + anotaciones de texto ──
    regimes_added_to_legend = set()
    price_max = float(df_aligned["High"].squeeze().max())
    price_min = float(df_aligned["Low"].squeeze().min())
    price_range = price_max - price_min
    annotation_y = price_max + price_range * 0.02

    i = 0
    while i < len(regimes):
        regime = regimes[i]
        start_idx = i
        while i < len(regimes) and regimes[i] == regime:
            i += 1
        end_idx = i - 1

        label = label_map[regime]
        color = color_map[regime]
        emoji = REGIME_EMOJIS.get(label, "")

        x0 = df_aligned.index[start_idx]
        x1 = df_aligned.index[min(end_idx, len(df_aligned.index) - 1)]

        # Fondo coloreado del regimen — mas visible (opacity 0.25)
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=0.25,
            layer="below",
            line_width=0,
        )

        # Anotacion de texto sobre cada bloque de regimen (solo si es suficientemente ancho)
        block_size = end_idx - start_idx + 1
        if block_size >= 5:
            mid_idx = start_idx + block_size // 2
            mid_x = df_aligned.index[mid_idx]
            fig.add_annotation(
                x=mid_x,
                y=annotation_y,
                text=f"<b>{emoji} {label}</b>",
                showarrow=False,
                font=dict(size=11, color=color),
                bgcolor="#131722",
                borderpad=3,
                opacity=0.9,
                yanchor="bottom",
            )

        # Leyenda
        if regime not in regimes_added_to_legend:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=14, color=color, symbol="square"),
                    name=f"{emoji} {label}",
                    showlegend=True,
                )
            )
            regimes_added_to_legend.add(regime)

    # ── Layout estilo TradingView ──
    fig.update_layout(
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color="#D1D4DC", size=13),
        height=700,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(
            gridcolor="#1E222D",
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#787B86",
            spikethickness=0.5,
            spikedash="dot",
            rangeslider_visible=False,
            color="#787B86",
        ),
        yaxis=dict(
            gridcolor="#1E222D",
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#787B86",
            spikethickness=0.5,
            spikedash="dot",
            side="right",
            color="#787B86",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color="#D1D4DC"),
            bgcolor="rgba(19,23,34,0.8)",
            bordercolor="#2A2E39",
            borderwidth=1,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1E222D",
            font_size=12,
            font_color="#D1D4DC",
            bordercolor="#2A2E39",
        ),
    )

    # Deshabilitar doble-click zoom reset para comportamiento mas TradingView
    fig.update_layout(xaxis_fixedrange=False, yaxis_fixedrange=False)

    return fig


def plot_transition_heatmap(transition_df: pd.DataFrame) -> go.Figure:
    """Heatmap de la matriz de transicion con probabilidades anotadas — estilo oscuro."""
    labels = transition_df.columns.tolist()
    z = transition_df.values

    text = [[f"{val:.2f}" for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 15, "color": "#D1D4DC"},
            colorscale=[
                [0, "#131722"],
                [0.25, "#1E3A5F"],
                [0.5, "#2962FF"],
                [0.75, "#448AFF"],
                [1, "#82B1FF"],
            ],
            showscale=True,
            zmin=0,
            zmax=1,
            colorbar=dict(
                tickfont=dict(color="#787B86"),
                title=dict(text="Prob.", font=dict(color="#787B86")),
            ),
        )
    )

    fig.update_layout(
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#D1D4DC", size=13),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title="Estado Destino (t+1)",
            color="#787B86",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Estado Origen (t)",
            autorange="reversed",
            color="#787B86",
            tickfont=dict(size=12),
        ),
    )

    return fig


def render_regime_card(current_label, current_color, ticker, timeframe, regime_means):
    """Renderiza el card principal del regimen actual con HTML custom."""
    emoji = REGIME_EMOJIS.get(current_label, "")
    st.html(
        f'<div class="regime-card" style="border-left:8px solid {current_color};">'
        f'<h2 style="color:{current_color};">{emoji} {current_label}</h2>'
        f'<p class="subtitle">{ticker} &mdash; {timeframe} &mdash; Regimen Actual</p>'
        f'<div class="metrics-row">'
        f'<div class="metric-item">'
        f'<span class="metric-label">Retorno Medio</span>'
        f'<span class="metric-value">{regime_means[0]:+.5f}</span></div>'
        f'<div class="metric-item">'
        f'<span class="metric-label">Volatilidad Media</span>'
        f'<span class="metric-value">{regime_means[1]:.5f}</span></div>'
        f'<div class="metric-item">'
        f'<span class="metric-label">Momentum Medio</span>'
        f'<span class="metric-value">{regime_means[2]:+.5f}</span></div>'
        f'</div></div>'
    )


def render_transition_probabilities(model, current_regime, label_map, color_map, sorted_order):
    """Renderiza las probabilidades de transicion con barras custom HTML."""
    st.html('<div class="section-title">Probabilidades de Transicion</div>')

    current_label = label_map[current_regime]
    emoji = REGIME_EMOJIS.get(current_label, "")
    st.html(
        f'<p style="color:#787B86;font-size:0.85rem;">Desde: <b style="color:{color_map[current_regime]};">'
        f'{emoji} {current_label}</b></p>'
    )

    trans_probs = model.transmat_[current_regime]
    bars_html = ""
    for regime_idx in sorted_order:
        prob = trans_probs[regime_idx]
        label = label_map[regime_idx]
        color = color_map[regime_idx]
        regime_emoji = REGIME_EMOJIS.get(label, "")
        pct_width = max(prob * 100, 1)

        bars_html += (
            f'<div class="prob-bar-container">'
            f'<div class="prob-bar-header">'
            f'<span class="prob-bar-label" style="color:{color};">{regime_emoji} {label}</span>'
            f'<span class="prob-bar-value" style="color:{color};">{prob:.1%}</span>'
            f'</div>'
            f'<div class="prob-bar-track">'
            f'<div class="prob-bar-fill" style="width:{pct_width}%;background:{color};"></div>'
            f'</div></div>'
        )

    st.html(bars_html)


def render_distribution_badges(regimes, label_map, color_map, sorted_order):
    """Renderiza la distribucion de regimenes como badges."""
    st.html('<div class="section-title">Distribucion de Regimenes</div>')

    badges_html = ""
    for regime_idx in sorted_order:
        count = int(np.sum(regimes == regime_idx))
        pct = count / len(regimes)
        label = label_map[regime_idx]
        color = color_map[regime_idx]
        emoji = REGIME_EMOJIS.get(label, "")
        badges_html += (
            f'<span class="dist-badge">'
            f'<span class="dot" style="background:{color};"></span>'
            f'{emoji} {label}: {pct:.1%} ({count})'
            f'</span>'
        )

    st.html(badges_html)


def render_transition_table(transition_df):
    """Renderiza la tabla de transicion como HTML sin dependencia de matplotlib."""
    labels = transition_df.columns.tolist()
    values = transition_df.values

    # Header
    header_cells = "".join(
        f'<th style="padding:10px 14px; color:#787B86; font-size:0.8rem; text-transform:uppercase; '
        f'letter-spacing:0.5px; border-bottom:1px solid #2A2E39;">{label}</th>'
        for label in labels
    )
    html = f"""
    <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; background:#1E222D; border-radius:8px; overflow:hidden;">
        <thead>
            <tr style="background:#131722;">
                <th style="padding:10px 14px; color:#787B86; border-bottom:1px solid #2A2E39;"></th>
                {header_cells}
            </tr>
        </thead>
        <tbody>
    """

    for i, row_label in enumerate(labels):
        row_html = (
            f'<td style="padding:10px 14px; color:#787B86; font-weight:600; font-size:0.85rem; '
            f'border-bottom:1px solid #2A2E39;">{row_label}</td>'
        )
        for j in range(len(labels)):
            val = values[i][j]
            # Color intensity based on value
            intensity = int(val * 200)
            bg_color = f"rgba(41, 98, 255, {val * 0.5})"
            text_color = "#D1D4DC" if val > 0.3 else "#787B86"
            font_weight = "700" if val > 0.5 else "400"
            row_html += (
                f'<td style="padding:10px 14px; text-align:center; font-family:Consolas,Monaco,monospace; '
                f'background:{bg_color}; color:{text_color}; font-weight:{font_weight}; font-size:0.95rem; '
                f'border-bottom:1px solid #2A2E39;">{val:.3f}</td>'
            )
        html += f"<tr>{row_html}</tr>"

    html += "</tbody></table></div>"
    st.html(html)


# ============================================================================
# SECTION F: Error Handling
# ============================================================================

def safe_train(features_scaled: np.ndarray, n_regimes: int):
    """
    Wrapper seguro para el entrenamiento del HMM.
    Returns: (model, converged, error_message)
    """
    try:
        model, converged = train_hmm(features_scaled, n_regimes)
        return model, converged, None
    except ValueError as e:
        return None, False, (
            f"No se pueden ajustar {n_regimes} regimenes a estos datos. "
            f"Intenta con menos regimenes. Detalle: {e}"
        )
    except Exception as e:
        return None, False, f"Error inesperado durante el entrenamiento: {e}"


# ============================================================================
# SECTION G: Streamlit App Layout
# ============================================================================

def main():
    st.set_page_config(
        page_title="Radar de Tendencia HMM",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inyectar CSS TradingView
    st.html(TRADINGVIEW_CSS)

    st.title("📡 Radar de Tendencia HMM")
    st.caption("Deteccion de regimenes de mercado con Hidden Markov Models")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.html(
            '<p style="color:#787B86;font-size:0.75rem;letter-spacing:1px;'
            'text-transform:uppercase;margin-bottom:16px;">Configuracion del Modelo</p>'
        )

        ticker = st.text_input(
            "Ticker / Simbolo",
            value="SPY",
            help="Acciones: SPY, AAPL, MSFT | Crypto: BTC/USDT, ETH/USDT",
        )

        timeframe = st.selectbox(
            "Timeframe",
            options=list(TIMEFRAME_CONFIG.keys()),
            format_func=lambda x: f"{x} ({TIMEFRAME_CONFIG[x]['label']})",
            index=2,
        )

        n_regimes = st.slider(
            "Regimenes HMM",
            min_value=3,
            max_value=7,
            value=3,
            help="Cantidad de estados ocultos del HMM (3=simple, 7=granular)",
        )

        st.divider()
        st.html(
            '<p style="color:#787B86;font-size:0.75rem;letter-spacing:1px;'
            'text-transform:uppercase;margin-bottom:8px;">Proyeccion</p>'
        )

        n_projection = st.slider(
            "Periodos hacia adelante (regimenes)",
            min_value=1,
            max_value=30,
            value=10,
            help="Cuantos periodos proyectar usando la matriz de transicion del HMM",
        )

        st.divider()
        st.html(
            '<p style="color:#FFD54F;font-size:0.75rem;letter-spacing:1px;'
            'text-transform:uppercase;margin-bottom:8px;">📐 Rango Grid / LP</p>'
        )

        range_horizon = st.number_input(
            "Horizonte del rango (periodos)",
            min_value=1,
            max_value=30,
            value=7,
            help="Periodos para proyectar el rango. Ej: 7 en 1D = 1 semana de rango",
        )

        z_score = st.select_slider(
            "Z-Score (bandas de confianza)",
            options=[1.0, 1.5, 2.0, 2.5, 3.0],
            value=2.0,
            help="1.5σ = ~87% confianza, 2σ = ~95%, 2.5σ = ~99%",
        )

        st.html("<br>")
        train_button = st.button("🚀 Entrenar Modelo", use_container_width=True)

        st.html("<br>")
        with st.expander("📐 Explicacion Matematica"):
            st.markdown(MATH_EXPLANATION)

    # ── Main Panel ───────────────────────────────────────────────────────

    if "trained" not in st.session_state:
        st.session_state.trained = False

    if train_button:
        _run_pipeline(ticker, timeframe, n_regimes, n_projection, range_horizon, z_score)

    if st.session_state.trained:
        _display_results()
    else:
        # Welcome screen
        st.html(
            '<div style="text-align:center;padding:80px 20px;">'
            '<p style="font-size:4rem;margin-bottom:16px;">📡</p>'
            '<h2 style="color:#D1D4DC;margin-bottom:8px;">Radar de Tendencia</h2>'
            '<p style="color:#787B86;font-size:1.1rem;max-width:500px;margin:0 auto;">'
            'Configura los parametros en el panel lateral y presiona '
            '<b style="color:#2962FF;">Entrenar Modelo</b> para detectar '
            'regimenes de mercado con HMM.</p></div>'
        )


def _run_pipeline(ticker: str, timeframe: str, n_regimes: int,
                  n_projection: int = 10, range_horizon: int = 7, z_score: float = 2.0):
    """Ejecuta el pipeline completo: fetch -> features -> train -> decode -> project -> range."""
    config = TIMEFRAME_CONFIG[timeframe]

    with st.spinner(f"Obteniendo datos para {ticker} ({timeframe})..."):
        try:
            df = fetch_data(ticker, timeframe)
        except ValueError as e:
            st.error(f"Error de datos: {e}")
            return
        except Exception as e:
            st.error(f"Error inesperado al obtener datos: {e}")
            return

    with st.spinner("Calculando features (retornos, volatilidad, momentum)..."):
        try:
            features = compute_features(df, config)
            features_scaled, scaler = normalize_features(features)
        except ValueError as e:
            st.error(f"Error en features: {e}")
            return

    with st.spinner(f"Entrenando HMM con {n_regimes} regimenes..."):
        model, converged, error_msg = safe_train(features_scaled, n_regimes)

        if error_msg:
            st.error(error_msg)
            return

        if not converged:
            st.warning(
                "El modelo no convergio completamente. Los resultados pueden "
                "ser menos confiables. Intenta reducir regimenes o usar un timeframe mas largo."
            )

    regimes = decode_regimes(model, features_scaled)
    label_map, color_map, sorted_order = label_regimes(model, n_regimes)
    transition_df = get_transition_matrix(model, label_map, sorted_order)

    # Store in session state
    st.session_state.model = model
    st.session_state.df = df
    st.session_state.features = features
    st.session_state.features_scaled = features_scaled
    st.session_state.regimes = regimes
    st.session_state.label_map = label_map
    st.session_state.color_map = color_map
    st.session_state.sorted_order = sorted_order
    st.session_state.transition_df = transition_df
    st.session_state.ticker = ticker
    st.session_state.timeframe = timeframe
    st.session_state.n_regimes = n_regimes
    st.session_state.n_projection = n_projection
    st.session_state.range_horizon = range_horizon
    st.session_state.z_score = z_score
    st.session_state.scaler = scaler
    st.session_state.converged = converged
    st.session_state.trained = True


def _display_results():
    """Muestra los resultados del modelo entrenado."""
    model = st.session_state.model
    df = st.session_state.df
    features = st.session_state.features
    regimes = st.session_state.regimes
    label_map = st.session_state.label_map
    color_map = st.session_state.color_map
    sorted_order = st.session_state.sorted_order
    transition_df = st.session_state.transition_df
    ticker = st.session_state.ticker
    timeframe = st.session_state.timeframe

    current_regime = regimes[-1]
    current_label = label_map[current_regime]
    current_color = color_map[current_regime]
    regime_means = model.means_[current_regime]
    scaler = st.session_state.scaler
    range_horizon = st.session_state.get("range_horizon", 7)
    z_score = st.session_state.get("z_score", 2.0)

    # ── 1. Card de regimen actual (full width, grande) ────────────────
    render_regime_card(current_label, current_color, ticker, timeframe, regime_means)

    # ── 1.5 Calculo y Card de Rango Grid/LP ──────────────────────────
    last_close = float(df["Close"].squeeze().iloc[-1])
    range_data = compute_range_projection(
        model, current_regime, last_close, range_horizon, z_score, scaler
    )
    render_range_card(range_data, current_label, current_color, timeframe)

    # ── 2. Grafico de velas + tunel de rango (full width, grande) ────
    fig_candle = plot_candlestick_with_regimes(
        df, regimes, label_map, color_map, features.index
    )
    # Agregar tunel de rango proyectado al chart
    last_date = features.index[-1]
    add_range_tunnel_to_chart(fig_candle, range_data, last_date, timeframe)

    st.plotly_chart(fig_candle, use_container_width=True, config={
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
        "displaylogo": False,
    })

    # ── 3. Probabilidades + Distribucion (debajo del chart) ──────────
    col_probs, col_dist = st.columns([1, 1])

    with col_probs:
        render_transition_probabilities(
            model, current_regime, label_map, color_map, sorted_order
        )

    with col_dist:
        render_distribution_badges(regimes, label_map, color_map, sorted_order)

        st.html("<br>")

        # Info de barras analizadas
        st.html(
            f'<div style="background:#1E222D;border-radius:8px;padding:12px 16px;'
            f'border:1px solid #2A2E39;margin-top:8px;">'
            f'<span style="color:#787B86;font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.5px;">Barras Analizadas</span><br>'
            f'<span style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
            f'font-size:1.4rem;font-weight:700;">{len(regimes):,}</span>'
            f'</div>'
        )

    # ── 4. Proyeccion de regimenes hacia adelante ──────────────────
    st.html("<br>")
    st.html('<div class="section-title">🔮 Proyeccion de Regimenes</div>')

    n_projection = st.session_state.get("n_projection", 10)
    proj_df = project_regimes(model, current_regime, n_projection, label_map, color_map, sorted_order)

    # Resumen: regimen mas probable en checkpoints clave
    render_projection_summary(proj_df, color_map, label_map, sorted_order, n_projection, timeframe)

    st.html("<br>")

    # Grafico de area apilada con evolucion de probabilidades
    fig_proj = plot_projection(proj_df, color_map, label_map, sorted_order, timeframe)
    st.plotly_chart(fig_proj, use_container_width=True, config={"displaylogo": False})

    # ── 5. Matriz de transicion ──────────────────────────────────────
    st.html("<br>")
    st.html('<div class="section-title">Matriz de Transicion</div>')

    col_heatmap, col_table = st.columns([3, 2])

    with col_heatmap:
        fig_heatmap = plot_transition_heatmap(transition_df)
        st.plotly_chart(fig_heatmap, use_container_width=True, config={"displaylogo": False})

    with col_table:
        st.html('<p style="color:#787B86;font-size:0.8rem;margin-bottom:8px;">Valores numericos:</p>')
        render_transition_table(transition_df)

    # ── 6. Features plot (debug) ─────────────────────────────────────
    with st.expander("📊 Features del Modelo (Debug / Analisis)"):
        fig_features = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Log Returns", "Volatilidad Realizada", "Momentum (EMA Slope)"],
            vertical_spacing=0.08,
        )

        fig_features.add_trace(
            go.Scatter(
                x=features.index, y=features["log_return"],
                name="Log Return", line=dict(color="#2962FF", width=1),
            ),
            row=1, col=1,
        )
        fig_features.add_trace(
            go.Scatter(
                x=features.index, y=features["volatility"],
                name="Volatilidad", line=dict(color="#FF6D00", width=1),
            ),
            row=2, col=1,
        )
        fig_features.add_trace(
            go.Scatter(
                x=features.index, y=features["momentum"],
                name="Momentum", line=dict(color="#AB47BC", width=1),
            ),
            row=3, col=1,
        )

        fig_features.update_layout(
            plot_bgcolor="#131722",
            paper_bgcolor="#131722",
            font=dict(color="#D1D4DC"),
            height=500,
            showlegend=False,
            margin=dict(l=60, r=20, t=40, b=20),
        )
        for axis_name in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
            fig_features.update_layout(**{
                axis_name: dict(gridcolor="#1E222D", zeroline=False, color="#787B86")
            })

        st.plotly_chart(fig_features, use_container_width=True, config={"displaylogo": False})


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
