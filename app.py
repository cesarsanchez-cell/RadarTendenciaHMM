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


def normalize_ticker(ticker: str) -> str:
    """
    Normaliza el ticker para evitar errores comunes:
    - Elimina espacios: 'btc / usdt' -> 'BTC/USDT'
    - Uppercase: 'btc/usdt' -> 'BTC/USDT'
    - Detecta pares sin '/' y los corrige: 'BTCUSDT' -> 'BTC/USDT'
    """
    ticker = ticker.strip().upper().replace(" ", "")

    # Si tiene '/' ya esta bien, solo limpiar
    if "/" in ticker:
        parts = ticker.split("/")
        return parts[0].strip() + "/" + parts[1].strip()

    # Detectar pares crypto sin '/' (ej: BTCUSDT, ETHUSDT)
    quote_currencies = ["USDT", "USDC", "BUSD", "USD", "BTC", "ETH", "BNB"]
    for quote in quote_currencies:
        if ticker.endswith(quote) and len(ticker) > len(quote):
            base = ticker[:-len(quote)]
            return base + "/" + quote

    # No es crypto, devolver como esta (accion/indice)
    return ticker


def search_similar_pairs(query: str, max_results: int = 10) -> list:
    """
    Busca pares en Binance que coincidan parcialmente con el query.
    Retorna lista de simbolos similares.
    """
    import ccxt
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        exchange.load_markets()
        query_upper = query.upper().replace(" ", "").replace("/", "")

        matches = []
        for symbol in exchange.symbols:
            symbol_clean = symbol.replace("/", "")
            # Match parcial: el query esta contenido en el simbolo
            if query_upper in symbol_clean:
                matches.append(symbol)
            # O la base del par coincide
            elif symbol.split("/")[0] == query_upper:
                matches.append(symbol)

        # Priorizar pares USDT, luego USDC, luego otros
        def sort_key(s):
            if s.endswith("/USDT"):
                return (0, s)
            elif s.endswith("/USDC"):
                return (1, s)
            else:
                return (2, s)

        matches.sort(key=sort_key)
        return matches[:max_results]
    except Exception:
        return []


def fetch_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """
    Obtiene datos OHLCV. Detecta si es crypto (contiene '/') para usar ccxt,
    sino usa yfinance. Aplica resample si el timeframe lo requiere.
    Normaliza el ticker automaticamente.
    """
    ticker = normalize_ticker(ticker)
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
# SECTION D.2: Monte Carlo Regime-Switching Range Projection
# ============================================================================

def _extract_regime_params(model: GaussianHMM, scaler):
    """
    Extrae mu y sigma de log-returns (feature 0) de cada regimen
    en escala original (invierte el RobustScaler).
    Returns: dict[regime_idx] -> (mu, sigma)
    """
    params = {}
    for k in range(model.n_components):
        mu_scaled = model.means_[k, 0]
        var_scaled = model.covars_[k][0, 0]
        # Invertir: x = x_scaled * IQR + median
        mu_orig = mu_scaled * scaler.scale_[0] + scaler.center_[0]
        sigma_orig = np.sqrt(var_scaled) * scaler.scale_[0]
        params[k] = (mu_orig, sigma_orig)
    return params


def compute_range_projection(model: GaussianHMM, current_regime: int,
                             last_close: float, horizon: int,
                             z_score: float, scaler,
                             n_simulations: int = 10000) -> dict:
    """
    Monte Carlo Regime-Switching: simula N trayectorias de precio donde
    en cada paso temporal:
      1. El regimen puede cambiar segun la matriz de transicion
      2. El retorno se samplea de la distribucion del regimen activo

    Esto produce rangos ASIMETRICOS y sesgados por la dinamica real
    de los regimenes (bear sesga abajo, bull sesga arriba).

    Returns dict con percentiles, tunnel por periodo, y estadisticas.
    """
    transmat = model.transmat_
    regime_params = _extract_regime_params(model, scaler)
    n_regimes = model.n_components

    # Pre-computar CDF acumulada de transicion para sampling rapido
    cum_transmat = np.cumsum(transmat, axis=1)

    # Matrices para almacenar resultados
    # prices[sim, step] = precio en ese paso de esa simulacion
    # regimes_sim[sim, step] = regimen activo
    prices = np.zeros((n_simulations, horizon + 1))
    prices[:, 0] = last_close

    # Estado inicial: todos arrancan en el regimen actual
    current_regimes = np.full(n_simulations, current_regime, dtype=int)

    # Random numbers pre-generados para velocidad
    rng = np.random.default_rng(seed=42)
    uniform_transitions = rng.random((n_simulations, horizon))
    uniform_returns = rng.standard_normal((n_simulations, horizon))

    # Almacenar regimenes por paso para estadisticas
    regime_counts_per_step = np.zeros((horizon, n_regimes))

    for t in range(horizon):
        # 1. Transicion de regimen para cada simulacion
        for sim in range(n_simulations):
            u = uniform_transitions[sim, t]
            cum_probs = cum_transmat[current_regimes[sim]]
            new_regime = np.searchsorted(cum_probs, u)
            new_regime = min(new_regime, n_regimes - 1)
            current_regimes[sim] = new_regime

        # Contar regimenes en este paso
        for k in range(n_regimes):
            regime_counts_per_step[t, k] = np.sum(current_regimes == k)

        # 2. Samplear retorno del regimen activo de cada simulacion
        for k in range(n_regimes):
            mask = current_regimes == k
            count = np.sum(mask)
            if count == 0:
                continue
            mu_k, sigma_k = regime_params[k]
            # r = mu + sigma * Z
            returns_k = mu_k + sigma_k * uniform_returns[mask, t]
            prices[mask, t + 1] = prices[mask, t] * np.exp(returns_k)

    # ── Calcular percentiles por paso ──
    # Convertir z_score a percentiles
    # z=1.5 -> p5/p95 (~90% CI), z=2.0 -> p2.5/p97.5 (~95% CI)
    from scipy.stats import norm
    lower_pct = (1 - norm.cdf(z_score)) * 100    # ej: 2.28 para z=2
    upper_pct = norm.cdf(z_score) * 100           # ej: 97.72 para z=2

    tunnel_upper = []
    tunnel_lower = []
    tunnel_mid = []        # mediana (p50)
    tunnel_expected = []   # media
    tunnel_p10 = []        # P10 (rango operativo 80%)
    tunnel_p25 = []
    tunnel_p75 = []
    tunnel_p90 = []        # P90 (rango operativo 80%)

    for t in range(1, horizon + 1):
        step_prices = prices[:, t]
        tunnel_lower.append(float(np.percentile(step_prices, lower_pct)))
        tunnel_p10.append(float(np.percentile(step_prices, 10)))
        tunnel_p25.append(float(np.percentile(step_prices, 25)))
        tunnel_mid.append(float(np.percentile(step_prices, 50)))
        tunnel_p75.append(float(np.percentile(step_prices, 75)))
        tunnel_p90.append(float(np.percentile(step_prices, 90)))
        tunnel_upper.append(float(np.percentile(step_prices, upper_pct)))
        tunnel_expected.append(float(np.mean(step_prices)))

    # Precios finales
    final_prices = prices[:, -1]
    upper_price = float(np.percentile(final_prices, upper_pct))
    lower_price = float(np.percentile(final_prices, lower_pct))
    mid_price = float(np.median(final_prices))
    expected_price = float(np.mean(final_prices))
    amplitude_pct = (upper_price - lower_price) / last_close * 100

    # Sesgo: que tan asimetrico es el rango
    upside = (upper_price - last_close) / last_close * 100
    downside = (last_close - lower_price) / last_close * 100
    skew_ratio = upside / downside if downside > 0 else float('inf')

    # Probabilidades direccionales
    prob_up = float(np.mean(final_prices > last_close)) * 100
    prob_down = float(np.mean(final_prices < last_close)) * 100

    # IQR range (50% de las trayectorias)
    iqr_upper = float(np.percentile(final_prices, 75))
    iqr_lower = float(np.percentile(final_prices, 25))
    iqr_amplitude = (iqr_upper - iqr_lower) / last_close * 100

    # Rango operativo 80% (P10-P90)
    op80_upper = float(np.percentile(final_prices, 90))
    op80_lower = float(np.percentile(final_prices, 10))
    op80_amplitude = (op80_upper - op80_lower) / last_close * 100

    # Regimen dominante promedio
    regime_probs_avg = regime_counts_per_step.mean(axis=0) / n_simulations

    return {
        "upper": upper_price,
        "lower": lower_price,
        "mid": mid_price,
        "expected": expected_price,
        "amplitude_pct": amplitude_pct,
        "upside_pct": upside,
        "downside_pct": downside,
        "skew_ratio": skew_ratio,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "iqr_upper": iqr_upper,
        "iqr_lower": iqr_lower,
        "iqr_amplitude": iqr_amplitude,
        "op80_upper": op80_upper,
        "op80_lower": op80_lower,
        "op80_amplitude": op80_amplitude,
        "z_score": z_score,
        "horizon": horizon,
        "last_close": last_close,
        "n_simulations": n_simulations,
        "tunnel_upper": tunnel_upper,
        "tunnel_lower": tunnel_lower,
        "tunnel_mid": tunnel_mid,
        "tunnel_expected": tunnel_expected,
        "tunnel_p10": tunnel_p10,
        "tunnel_p25": tunnel_p25,
        "tunnel_p75": tunnel_p75,
        "tunnel_p90": tunnel_p90,
        "regime_probs_avg": regime_probs_avg,
    }


def add_range_tunnel_to_chart(fig: go.Figure, range_data: dict,
                               last_date, timeframe: str):
    """
    Agrega el tunel Monte Carlo al grafico de velas.
    Muestra 2 zonas: IQR (50% trayectorias) + banda completa (z-score).
    La mediana muestra hacia donde sesga el modelo.
    """
    horizon = range_data["horizon"]
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

    all_dates = [last_date] + future_dates
    all_upper = [last_close] + range_data["tunnel_upper"]
    all_lower = [last_close] + range_data["tunnel_lower"]
    all_p90 = [last_close] + range_data["tunnel_p90"]
    all_p75 = [last_close] + range_data["tunnel_p75"]
    all_p25 = [last_close] + range_data["tunnel_p25"]
    all_p10 = [last_close] + range_data["tunnel_p10"]
    all_mid = [last_close] + range_data["tunnel_mid"]

    # Banda exterior (z-score CI ~95%)
    fig.add_trace(
        go.Scatter(
            x=all_dates, y=all_upper, mode="lines",
            line=dict(color="#FFD54F", width=1, dash="dot"),
            name=f"~95% CI",
            showlegend=True,
            hovertemplate="P97.5: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates, y=all_lower, mode="lines",
            line=dict(color="#FFD54F", width=1, dash="dot"),
            name=f"~95% CI Lower",
            fill="tonexty",
            fillcolor="rgba(255, 213, 79, 0.04)",
            showlegend=False,
            hovertemplate="P2.5: $%{y:,.2f}<extra></extra>",
        )
    )

    # Banda operativa 80% (P10-P90) — EL RANGO CLAVE
    fig.add_trace(
        go.Scatter(
            x=all_dates, y=all_p90, mode="lines",
            line=dict(color="#AB47BC", width=2, dash="dash"),
            name="Rango 80% (P90)",
            showlegend=True,
            hovertemplate="P90: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates, y=all_p10, mode="lines",
            line=dict(color="#AB47BC", width=2, dash="dash"),
            name="Rango 80% (P10)",
            fill="tonexty",
            fillcolor="rgba(171, 71, 188, 0.10)",
            showlegend=True,
            hovertemplate="P10: $%{y:,.2f}<extra></extra>",
        )
    )

    # Banda interior IQR (P25-P75, 50%)
    fig.add_trace(
        go.Scatter(
            x=all_dates, y=all_p75, mode="lines",
            line=dict(color="#42A5F5", width=1.5),
            name="P75 (IQR 50%)",
            showlegend=True,
            hovertemplate="P75: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=all_dates, y=all_p25, mode="lines",
            line=dict(color="#42A5F5", width=1.5),
            name="P25 (IQR 50%)",
            fill="tonexty",
            fillcolor="rgba(66, 165, 245, 0.10)",
            showlegend=True,
            hovertemplate="P25: $%{y:,.2f}<extra></extra>",
        )
    )

    # Mediana (hacia donde sesga el modelo)
    fig.add_trace(
        go.Scatter(
            x=all_dates, y=all_mid, mode="lines",
            line=dict(color="#FFFFFF", width=2),
            name="Mediana (P50)",
            showlegend=True,
            hovertemplate="Mediana: $%{y:,.2f}<extra></extra>",
        )
    )

    # Anotaciones en el extremo derecho
    last_future = future_dates[-1]
    annotations = [
        (range_data["tunnel_upper"][-1], "#FFD54F", "left"),
        (range_data["tunnel_p90"][-1], "#AB47BC", "left"),
        (range_data["tunnel_p75"][-1], "#42A5F5", "left"),
        (range_data["tunnel_mid"][-1], "#FFFFFF", "left"),
        (range_data["tunnel_p25"][-1], "#42A5F5", "left"),
        (range_data["tunnel_p10"][-1], "#AB47BC", "left"),
        (range_data["tunnel_lower"][-1], "#FFD54F", "left"),
    ]
    for price, color, anchor in annotations:
        fig.add_annotation(
            x=last_future, y=price,
            text=f"<b>${price:,.0f}</b>",
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="#131722",
            xanchor=anchor, xshift=8,
        )

    return fig


def render_range_card(range_data: dict, current_label: str, current_color: str, timeframe: str):
    """Renderiza la tarjeta de rango Monte Carlo para Grid/LP."""
    tf_units = {"1H": "hora(s)", "4H": "periodo(s) 4H", "1D": "dia(s)", "1W": "semana(s)"}
    unit = tf_units.get(timeframe, "periodo(s)")

    upper = range_data["upper"]
    lower = range_data["lower"]
    mid = range_data["mid"]
    expected = range_data["expected"]
    amp = range_data["amplitude_pct"]
    iqr_amp = range_data["iqr_amplitude"]
    iqr_upper = range_data["iqr_upper"]
    iqr_lower = range_data["iqr_lower"]
    horizon = range_data["horizon"]
    z = range_data["z_score"]
    last = range_data["last_close"]
    prob_up = range_data["prob_up"]
    prob_down = range_data["prob_down"]
    skew = range_data["skew_ratio"]
    n_sim = range_data["n_simulations"]

    # Sesgo direccional
    if prob_up > 60:
        bias_color = "#4CAF50"
        bias_label = "Sesgo Alcista"
        bias_arrow = "↗"
    elif prob_down > 60:
        bias_color = "#EF5350"
        bias_label = "Sesgo Bajista"
        bias_arrow = "↘"
    else:
        bias_color = "#FFD54F"
        bias_label = "Sin Sesgo Claro"
        bias_arrow = "↔"

    # Amplitud IQR (rango operativo real)
    if iqr_amp < 5:
        iqr_color = "#4CAF50"
        iqr_label = "Estrecho"
    elif iqr_amp < 15:
        iqr_color = "#FFD54F"
        iqr_label = "Moderado"
    else:
        iqr_color = "#EF5350"
        iqr_label = "Amplio"

    st.html(
        f'<div style="background:linear-gradient(135deg,#1E222D 0%,#131722 100%);'
        f'border-radius:12px;padding:24px 28px;margin-bottom:16px;'
        f'border:1px solid #2A2E39;border-top:3px solid #FFD54F;'
        f'box-shadow:0 4px 24px rgba(0,0,0,0.3);">'
        # Header
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;flex-wrap:wrap;gap:12px;">'
        f'<div>'
        f'<p style="color:#FFD54F;font-size:0.75rem;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 4px 0;">📐 Monte Carlo Regime-Switching ({n_sim:,} sims)</p>'
        f'<p style="color:#787B86;font-size:0.8rem;margin:0;">'
        f'Horizonte: {horizon} {unit} | Regimen: {current_label}</p>'
        f'</div>'
        # Probabilidad direccional
        f'<div style="display:flex;gap:16px;align-items:center;">'
        f'<div style="text-align:center;">'
        f'<p style="color:{bias_color};font-size:1.6rem;font-weight:700;'
        f'font-family:Consolas,Monaco,monospace;margin:0;">{bias_arrow} {prob_up:.0f}%↑ / {prob_down:.0f}%↓</p>'
        f'<p style="color:{bias_color};font-size:0.7rem;margin:0;">{bias_label}</p>'
        f'</div></div></div>'
        # Rangos: 80% (operativo) + IQR (50%) + Full (~95%)
        f'<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">'
        # 80% Box — EL RANGO CLAVE
        f'<div style="flex:1.3;min-width:220px;background:#131722;border-radius:8px;'
        f'padding:14px 16px;border:2px solid #AB47BC;">'
        f'<p style="color:#AB47BC;font-size:0.75rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 8px 0;font-weight:700;">'
        f'🎯 Rango Operativo (80% prob)</p>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<span style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.3rem;font-weight:700;">${range_data["op80_lower"]:,.2f} — ${range_data["op80_upper"]:,.2f}</span>'
        f'<span style="color:#AB47BC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.2rem;font-weight:700;">{range_data["op80_amplitude"]:.1f}%</span>'
        f'</div></div>'
        # IQR Box (50%)
        f'<div style="flex:1;min-width:200px;background:#131722;border-radius:8px;'
        f'padding:14px 16px;border:1px solid #42A5F5;">'
        f'<p style="color:#42A5F5;font-size:0.7rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 8px 0;">Rango 50% (IQR)</p>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<span style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.1rem;">${iqr_lower:,.2f} — ${iqr_upper:,.2f}</span>'
        f'<span style="color:{iqr_color};font-family:Consolas,Monaco,monospace;'
        f'font-size:1rem;">{iqr_amp:.1f}%</span>'
        f'</div></div>'
        # Full range Box (~95%)
        f'<div style="flex:1;min-width:200px;background:#131722;border-radius:8px;'
        f'padding:14px 16px;border:1px solid #FFD54F44;">'
        f'<p style="color:#FFD54F;font-size:0.7rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 8px 0;">Rango Extremo (~95%)</p>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<span style="color:#787B86;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.1rem;">${lower:,.2f} — ${upper:,.2f}</span>'
        f'<span style="color:#787B86;font-family:Consolas,Monaco,monospace;'
        f'font-size:1rem;">{amp:.1f}%</span>'
        f'</div></div></div>'
        # Grid de precios detallado
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
        # Mediana
        f'<div style="flex:1;min-width:110px;background:#131722;border-radius:8px;'
        f'padding:10px 14px;border:1px solid #2A2E39;">'
        f'<p style="color:#FFFFFF;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 3px 0;">Mediana (P50)</p>'
        f'<p style="color:#FFFFFF;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.1rem;font-weight:700;margin:0;">${mid:,.2f}</p>'
        f'<p style="color:#787B86;font-size:0.75rem;margin:2px 0 0 0;">'
        f'{((mid/last)-1)*100:+.2f}%</p></div>'
        # Upside
        f'<div style="flex:1;min-width:110px;background:#131722;border-radius:8px;'
        f'padding:10px 14px;border:1px solid #2A2E39;">'
        f'<p style="color:#EF5350;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 3px 0;">Upside max</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.1rem;font-weight:700;margin:0;">+{range_data["upside_pct"]:.1f}%</p></div>'
        # Downside
        f'<div style="flex:1;min-width:110px;background:#131722;border-radius:8px;'
        f'padding:10px 14px;border:1px solid #2A2E39;">'
        f'<p style="color:#4CAF50;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 3px 0;">Downside max</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.1rem;font-weight:700;margin:0;">-{range_data["downside_pct"]:.1f}%</p></div>'
        # Precio actual
        f'<div style="flex:1;min-width:110px;background:#131722;border-radius:8px;'
        f'padding:10px 14px;border:1px solid #2A2E39;">'
        f'<p style="color:#787B86;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 3px 0;">Precio Actual</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:1.1rem;font-weight:700;margin:0;">${last:,.2f}</p></div>'
        f'</div></div>'
    )


def render_strategy_recommendation(range_data: dict, current_label: str,
                                    current_color: str, timeframe: str):
    """
    Recomienda LP concentrada vs Grid vs Hedge basado en:
    - Regimen actual (tendencia/fuerza)
    - Amplitud del rango operativo 80%
    - Sesgo direccional (prob_up vs prob_down)
    """
    op80_amp = range_data["op80_amplitude"]
    prob_up = range_data["prob_up"]
    prob_down = range_data["prob_down"]
    op80_upper = range_data["op80_upper"]
    op80_lower = range_data["op80_lower"]
    last = range_data["last_close"]
    mid = range_data["mid"]
    skew = range_data["skew_ratio"]

    # Clasificar regimen
    label_lower = current_label.lower()
    is_strong = "strong" in label_lower
    is_sideways = "sideways" in label_lower
    is_bull = "bull" in label_lower
    is_bear = "bear" in label_lower

    # Clasificar amplitud
    narrow = op80_amp < 12
    moderate = 12 <= op80_amp <= 25
    wide = op80_amp > 25

    # Determinar recomendacion
    if is_sideways and narrow:
        strategy = "LP CONCENTRADA"
        strategy_color = "#4CAF50"
        strategy_icon = "🎯"
        reason = "Rango estrecho + sin tendencia = maximo fee generation"
        details = (
            f"Posicionar LP en [{op80_lower:,.0f} — {op80_upper:,.0f}] "
            f"con concentracion alta. Amplitud {op80_amp:.1f}% es ideal para "
            f"capturar fees sin sufrir IL significativo."
        )
        risk = "Bajo. Monitorear si ADX sube >25 (inicio de tendencia)."
        grid_levels = "N/A — LP concentrada es mas eficiente en este escenario"

    elif is_sideways and moderate:
        strategy = "GRID SIMETRICO"
        strategy_color = "#42A5F5"
        strategy_icon = "📊"
        reason = "Rango moderado + sin tendencia = cada rebote es profit"
        levels = max(8, min(20, int(op80_amp / 1.5)))
        spacing = op80_amp / levels
        details = (
            f"Grid de {levels} niveles entre [{op80_lower:,.0f} — {op80_upper:,.0f}]. "
            f"Spacing: ~{spacing:.1f}% entre niveles. "
            f"Centrado en precio actual (${last:,.0f})."
        )
        risk = "Moderado. Si rompe el rango, el grid pierde inventario en un lado."
        grid_levels = f"{levels} niveles | Spacing: {spacing:.1f}%"

    elif (is_bull or is_bear) and not is_strong and moderate:
        strategy = "GRID ASIMETRICO"
        strategy_color = "#FFD54F"
        strategy_icon = "📐"
        if is_bull:
            bias_dir = "alcista"
            more_levels_side = "arriba"
            # Mas niveles arriba del precio actual
            levels_up = 8
            levels_down = 5
        else:
            bias_dir = "bajista"
            more_levels_side = "abajo"
            levels_up = 5
            levels_down = 8

        total_levels = levels_up + levels_down
        reason = f"Tendencia {bias_dir} moderada = sesgar grid hacia {more_levels_side}"
        details = (
            f"Grid asimetrico: {levels_up} niveles arriba + {levels_down} abajo del precio actual. "
            f"Rango [{op80_lower:,.0f} — {op80_upper:,.0f}]. "
            f"Mas niveles en la direccion del sesgo para capturar el movimiento."
        )
        risk = f"Moderado-Alto. Si revierte contra el sesgo, perdida de inventario."
        grid_levels = f"{total_levels} niveles ({levels_up} up / {levels_down} down)"

    elif is_strong or wide:
        strategy = "NO OPERAR / HEDGE PURO"
        strategy_color = "#EF5350"
        strategy_icon = "🛑"
        reason = "Tendencia fuerte o rango muy amplio = IL destruye LP, grid pierde inventario"
        details = (
            f"Amplitud {op80_amp:.1f}% con regimen '{current_label}'. "
            f"El riesgo de impermanent loss o perdida de inventario es demasiado alto. "
            f"Mantener hedge delta-neutral puro hasta que el regimen cambie a Sideways o la vol baje."
        )
        risk = "Alto. Esperar cambio de regimen antes de posicionar."
        grid_levels = "N/A — No se recomienda operar grid/LP"

    elif is_sideways and wide:
        strategy = "GRID AMPLIO CONSERVADOR"
        strategy_color = "#FF9800"
        strategy_icon = "⚠️"
        reason = "Sin tendencia pero vol alta = grid con spacing amplio"
        levels = max(6, min(12, int(op80_amp / 3)))
        spacing = op80_amp / levels
        details = (
            f"Grid conservador de {levels} niveles con spacing amplio (~{spacing:.1f}%). "
            f"Rango [{op80_lower:,.0f} — {op80_upper:,.0f}]. "
            f"Menos niveles para reducir exposicion ante la alta volatilidad."
        )
        risk = "Moderado-Alto. Vol alta puede generar movimientos bruscos."
        grid_levels = f"{levels} niveles | Spacing: {spacing:.1f}%"

    else:
        # Default: moderado sin tendencia clara
        strategy = "GRID NEUTRAL"
        strategy_color = "#78909C"
        strategy_icon = "⚖️"
        levels = 10
        spacing = op80_amp / levels if op80_amp > 0 else 1
        reason = "Condiciones mixtas — grid neutral como posicion base"
        details = (
            f"Grid de {levels} niveles en [{op80_lower:,.0f} — {op80_upper:,.0f}]. "
            f"Monitorear regimen para ajustar."
        )
        risk = "Moderado."
        grid_levels = f"{levels} niveles | Spacing: {spacing:.1f}%"

    # Posicion del precio dentro del rango 80%
    if op80_upper > op80_lower:
        position_in_range = (last - op80_lower) / (op80_upper - op80_lower) * 100
    else:
        position_in_range = 50

    if position_in_range > 75:
        pos_label = "Cerca del techo"
        pos_color = "#EF5350"
    elif position_in_range < 25:
        pos_label = "Cerca del piso"
        pos_color = "#4CAF50"
    else:
        pos_label = "Zona central"
        pos_color = "#78909C"

    st.html(
        f'<div style="background:linear-gradient(135deg,#1E222D 0%,#131722 100%);'
        f'border-radius:12px;padding:24px 28px;margin-bottom:16px;'
        f'border:1px solid #2A2E39;border-top:3px solid {strategy_color};'
        f'box-shadow:0 4px 24px rgba(0,0,0,0.3);">'
        # Header
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:16px;flex-wrap:wrap;gap:12px;">'
        f'<div>'
        f'<p style="color:{strategy_color};font-size:0.75rem;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 4px 0;">RECOMENDACION DE ESTRATEGIA</p>'
        f'<p style="color:{strategy_color};font-size:1.8rem;font-weight:700;'
        f'font-family:Consolas,Monaco,monospace;margin:0;">'
        f'{strategy_icon} {strategy}</p>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<p style="color:#787B86;font-size:0.7rem;margin:0 0 2px 0;">Regimen: {current_label}</p>'
        f'<p style="color:#787B86;font-size:0.7rem;margin:0;">Amplitud 80%: {op80_amp:.1f}%</p>'
        f'</div></div>'
        # Razon
        f'<div style="background:#131722;border-radius:8px;padding:12px 16px;'
        f'margin-bottom:12px;border-left:3px solid {strategy_color};">'
        f'<p style="color:#D1D4DC;font-size:0.9rem;margin:0;line-height:1.5;">'
        f'<b>Por que:</b> {reason}</p></div>'
        # Detalles
        f'<div style="background:#131722;border-radius:8px;padding:12px 16px;'
        f'margin-bottom:12px;border:1px solid #2A2E39;">'
        f'<p style="color:#D1D4DC;font-size:0.85rem;margin:0;line-height:1.5;">'
        f'{details}</p></div>'
        # Grid de metricas
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
        # Grid levels
        f'<div style="flex:1;min-width:150px;background:#131722;border-radius:8px;'
        f'padding:10px 14px;border:1px solid #2A2E39;">'
        f'<p style="color:#787B86;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 3px 0;">Grid Config</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:0.9rem;margin:0;">{grid_levels}</p></div>'
        # Position in range
        f'<div style="flex:1;min-width:150px;background:#131722;border-radius:8px;'
        f'padding:10px 14px;border:1px solid #2A2E39;">'
        f'<p style="color:#787B86;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 3px 0;">Posicion en Rango 80%</p>'
        f'<p style="color:{pos_color};font-family:Consolas,Monaco,monospace;'
        f'font-size:0.9rem;margin:0;">{pos_label} ({position_in_range:.0f}%)</p></div>'
        # Risk
        f'<div style="flex:1;min-width:150px;background:#131722;border-radius:8px;'
        f'padding:10px 14px;border:1px solid #2A2E39;">'
        f'<p style="color:#787B86;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 3px 0;">Riesgo</p>'
        f'<p style="color:#D1D4DC;font-size:0.85rem;margin:0;">{risk}</p></div>'
        # Sesgo
        f'<div style="flex:1;min-width:150px;background:#131722;border-radius:8px;'
        f'padding:10px 14px;border:1px solid #2A2E39;">'
        f'<p style="color:#787B86;font-size:0.65rem;text-transform:uppercase;'
        f'letter-spacing:0.8px;margin:0 0 3px 0;">Sesgo Direccional</p>'
        f'<p style="color:#D1D4DC;font-family:Consolas,Monaco,monospace;'
        f'font-size:0.9rem;margin:0;">{prob_up:.0f}% UP / {prob_down:.0f}% DN</p></div>'
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

        # Pares populares para selector rapido
        POPULAR_CRYPTO = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT",
            "MATIC/USDT", "UNI/USDT", "AAVE/USDT", "ARB/USDT", "OP/USDT",
        ]
        POPULAR_STOCKS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL"]

        asset_type = st.radio(
            "Tipo de activo",
            ["Crypto", "Acciones/Indices"],
            horizontal=True,
        )

        if asset_type == "Crypto":
            col_select, col_custom = st.columns([2, 1])
            with col_select:
                ticker = st.selectbox(
                    "Par (populares)",
                    options=POPULAR_CRYPTO,
                    index=0,
                    help="Selecciona un par popular o escribe uno custom",
                )
            with col_custom:
                custom_ticker = st.text_input(
                    "Custom",
                    value="",
                    placeholder="ej: PEPE/USDT",
                    help="Escribe cualquier par. Se normaliza automaticamente",
                )
            if custom_ticker.strip():
                ticker = normalize_ticker(custom_ticker)

            # Buscar pares similares si el custom no esta vacio
            if custom_ticker.strip() and "/" not in custom_ticker and len(custom_ticker) >= 2:
                with st.spinner("Buscando pares..."):
                    similar = search_similar_pairs(custom_ticker)
                if similar:
                    ticker = st.selectbox(
                        "Pares encontrados",
                        options=similar,
                        index=0,
                    )
        else:
            ticker = st.selectbox(
                "Ticker",
                options=POPULAR_STOCKS,
                index=0,
                help="Selecciona un ticker o escribe uno custom abajo",
            )
            custom_stock = st.text_input(
                "Custom ticker",
                value="",
                placeholder="ej: META, AMD",
            )
            if custom_stock.strip():
                ticker = custom_stock.strip().upper()

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

    # ── 1.6 Recomendacion de estrategia LP/Grid/Hedge ────────────────
    render_strategy_recommendation(range_data, current_label, current_color, timeframe)

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
