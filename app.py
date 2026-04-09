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
    "30m": {
        "yf_interval": "30m",
        "yf_period": "60d",
        "resample_rule": None,
        "vol_window": 20,
        "momentum_ema_span": 14,
        "momentum_slope_window": 5,
        "min_bars_required": 100,
        "label": "30 Minutos",
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
        tf_map = {"30m": "30m", "4H": "1h", "1D": "1d", "1W": "1d"}
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
    st.markdown(
        f"""
        <div class="regime-card" style="border-left: 8px solid {current_color};">
            <h2 style="color: {current_color};">{emoji} {current_label}</h2>
            <p class="subtitle">{ticker} &mdash; {timeframe} &mdash; Regimen Actual</p>
            <div class="metrics-row">
                <div class="metric-item">
                    <span class="metric-label">Retorno Medio</span>
                    <span class="metric-value">{regime_means[0]:+.5f}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Volatilidad Media</span>
                    <span class="metric-value">{regime_means[1]:.5f}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Momentum Medio</span>
                    <span class="metric-value">{regime_means[2]:+.5f}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_transition_probabilities(model, current_regime, label_map, color_map, sorted_order):
    """Renderiza las probabilidades de transicion con barras custom HTML."""
    st.markdown('<div class="section-title">Probabilidades de Transicion</div>', unsafe_allow_html=True)

    current_label = label_map[current_regime]
    emoji = REGIME_EMOJIS.get(current_label, "")
    st.markdown(
        f'<p style="color: #787B86; font-size: 0.85rem;">Desde: <b style="color: {color_map[current_regime]};">'
        f'{emoji} {current_label}</b></p>',
        unsafe_allow_html=True,
    )

    trans_probs = model.transmat_[current_regime]
    for regime_idx in sorted_order:
        prob = trans_probs[regime_idx]
        label = label_map[regime_idx]
        color = color_map[regime_idx]
        regime_emoji = REGIME_EMOJIS.get(label, "")
        pct_width = max(prob * 100, 1)  # min 1% para visibilidad

        st.markdown(
            f"""
            <div class="prob-bar-container">
                <div class="prob-bar-header">
                    <span class="prob-bar-label" style="color: {color};">{regime_emoji} {label}</span>
                    <span class="prob-bar-value" style="color: {color};">{prob:.1%}</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: {pct_width}%; background: {color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_distribution_badges(regimes, label_map, color_map, sorted_order):
    """Renderiza la distribucion de regimenes como badges."""
    st.markdown('<div class="section-title">Distribucion de Regimenes</div>', unsafe_allow_html=True)

    badges_html = ""
    for regime_idx in sorted_order:
        count = int(np.sum(regimes == regime_idx))
        pct = count / len(regimes)
        label = label_map[regime_idx]
        color = color_map[regime_idx]
        emoji = REGIME_EMOJIS.get(label, "")
        badges_html += (
            f'<span class="dist-badge">'
            f'<span class="dot" style="background: {color};"></span>'
            f'{emoji} {label}: {pct:.1%} ({count})'
            f'</span>'
        )

    st.markdown(badges_html, unsafe_allow_html=True)


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
    st.markdown(html, unsafe_allow_html=True)


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
    st.markdown(TRADINGVIEW_CSS, unsafe_allow_html=True)

    st.title("📡 Radar de Tendencia HMM")
    st.caption("Deteccion de regimenes de mercado con Hidden Markov Models")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<p style="color:#787B86; font-size:0.75rem; letter-spacing:1px; '
            'text-transform:uppercase; margin-bottom:16px;">Configuracion del Modelo</p>',
            unsafe_allow_html=True,
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

        st.markdown("<br>", unsafe_allow_html=True)
        train_button = st.button("🚀 Entrenar Modelo", use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.expander("📐 Explicacion Matematica"):
            st.markdown(MATH_EXPLANATION)

    # ── Main Panel ───────────────────────────────────────────────────────

    if "trained" not in st.session_state:
        st.session_state.trained = False

    if train_button:
        _run_pipeline(ticker, timeframe, n_regimes)

    if st.session_state.trained:
        _display_results()
    else:
        # Welcome screen
        st.markdown(
            """
            <div style="text-align:center; padding:80px 20px;">
                <p style="font-size:4rem; margin-bottom:16px;">📡</p>
                <h2 style="color:#D1D4DC; margin-bottom:8px;">Radar de Tendencia</h2>
                <p style="color:#787B86; font-size:1.1rem; max-width:500px; margin:0 auto;">
                    Configura los parametros en el panel lateral y presiona
                    <b style="color:#2962FF;">Entrenar Modelo</b> para detectar
                    regimenes de mercado con HMM.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _run_pipeline(ticker: str, timeframe: str, n_regimes: int):
    """Ejecuta el pipeline completo: fetch -> features -> train -> decode."""
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

    # ── 1. Card de regimen actual (full width, grande) ────────────────
    render_regime_card(current_label, current_color, ticker, timeframe, regime_means)

    # ── 2. Grafico de velas (full width, grande) ─────────────────────
    fig_candle = plot_candlestick_with_regimes(
        df, regimes, label_map, color_map, features.index
    )
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

        st.markdown("<br>", unsafe_allow_html=True)

        # Info de barras analizadas
        st.markdown(
            f'<div style="background:#1E222D; border-radius:8px; padding:12px 16px; '
            f'border:1px solid #2A2E39; margin-top:8px;">'
            f'<span style="color:#787B86; font-size:0.8rem; text-transform:uppercase; '
            f'letter-spacing:0.5px;">Barras Analizadas</span><br>'
            f'<span style="color:#D1D4DC; font-family:Consolas,Monaco,monospace; '
            f'font-size:1.4rem; font-weight:700;">{len(regimes):,}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 4. Matriz de transicion ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matriz de Transicion</div>', unsafe_allow_html=True)

    col_heatmap, col_table = st.columns([3, 2])

    with col_heatmap:
        fig_heatmap = plot_transition_heatmap(transition_df)
        st.plotly_chart(fig_heatmap, use_container_width=True, config={"displaylogo": False})

    with col_table:
        st.markdown(
            '<p style="color:#787B86; font-size:0.8rem; margin-bottom:8px;">Valores numericos:</p>',
            unsafe_allow_html=True,
        )
        render_transition_table(transition_df)

    # ── 5. Features plot (debug) ─────────────────────────────────────
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
