"""
Radar de Tendencia HMM
======================
Monitor de mercado con detección de regímenes usando Hidden Markov Models.
Aplicación Streamlit con GaussianHMM multivariado, candlestick interactivo
y matriz de transición.

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
        "label": "1 Día",
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

# Paleta de colores por régimen: de bear (rojo) a bull (verde)
# Se seleccionan según la cantidad de regímenes
REGIME_PALETTES = {
    3: ["#d32f2f", "#9e9e9e", "#2e7d32"],
    4: ["#d32f2f", "#ef5350", "#66bb6a", "#2e7d32"],
    5: ["#d32f2f", "#ef5350", "#9e9e9e", "#66bb6a", "#2e7d32"],
    6: ["#b71c1c", "#d32f2f", "#ef5350", "#66bb6a", "#2e7d32", "#1b5e20"],
    7: ["#b71c1c", "#d32f2f", "#ef5350", "#9e9e9e", "#66bb6a", "#2e7d32", "#1b5e20"],
}

# Labels semánticos por cantidad de regímenes
REGIME_LABELS = {
    3: ["Bear", "Sideways", "Bull"],
    4: ["Strong Bear", "Bear", "Bull", "Strong Bull"],
    5: ["Strong Bear", "Bear", "Sideways", "Bull", "Strong Bull"],
    6: ["Strong Bear", "Bear", "Mild Bear", "Mild Bull", "Bull", "Strong Bull"],
    7: ["Strong Bear", "Bear", "Mild Bear", "Sideways", "Mild Bull", "Bull", "Strong Bull"],
}

MATH_EXPLANATION = r"""
### Enfoque Matemático

**Hidden Markov Model (HMM)** — El mercado se modela como un sistema con *K* estados
ocultos (no observables), donde cada estado representa un **régimen de mercado** distinto.
En cada paso temporal, el sistema ocupa un estado y emite un vector de observación
extraído de una distribución Gaussiana multivariada específica de ese estado.

**Formalmente:**
- Estados ocultos: $S = \{s_1, ..., s_K\}$ (los regímenes)
- Matriz de transición: $A$ donde $A_{ij} = P(state_t = s_j \mid state_{t-1} = s_i)$
- Distribuciones de emisión: cada estado $s_k$ tiene parámetros $(\mu_k, \Sigma_k)$

**El vector de observación** tiene 3 dimensiones:

$$\mathbf{x}_t = [r_t, \sigma_t, m_t]$$

1. **Retorno logarítmico** $r_t = \ln(C_t / C_{t-1})$ — Captura la dirección del precio.
   Son aditivos en el tiempo y aproximadamente normales para valores pequeños, lo que
   se alinea con la suposición Gaussiana del HMM.

2. **Volatilidad realizada** $\sigma_t = \text{std}(r_{t-w+1}, ..., r_t)$ — Desviación estándar
   móvil de los retornos sobre una ventana $w$. Separa regímenes de alta/baja volatilidad
   (crisis vs. tendencias estables).

3. **Momentum direccional** — Pendiente de la EMA:
   $m_t = \frac{EMA_t - EMA_{t-k}}{k}$, donde $EMA_t = \alpha \cdot C_t + (1-\alpha) \cdot EMA_{t-1}$
   y $\alpha = 2/(span+1)$. Es continuo e ilimitado, ideal para emisiones Gaussianas.

**Entrenamiento:** Algoritmo Baum-Welch (Expectation-Maximization) para estimar $A$, $\mu_k$ y $\Sigma_k$.

**Decodificación:** Algoritmo de Viterbi para la secuencia más probable de estados ocultos.

**Normalización:** Se usa `RobustScaler` (mediana + IQR) en vez de `StandardScaler`
para ser robusto a los fat tails típicos de datos financieros.

**Covarianza completa** (`covariance_type="full"`) captura correlaciones entre features,
como el *leverage effect* (retornos negativos ↔ volatilidad creciente).
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
    # Filtrar solo columnas que existen
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

    # Resamplear si es necesario (4H, 1W)
    if config["resample_rule"] is not None:
        df = resample_ohlcv(df, config["resample_rule"])

    # Validar mínimo de barras
    if len(df) < config["min_bars_required"]:
        raise ValueError(
            f"Solo {len(df)} barras disponibles. Se requieren al menos "
            f"{config['min_bars_required']} para el timeframe {timeframe}. "
            f"Intenta con un timeframe más largo o un ticker con más historia."
        )

    return df


def _fetch_traditional(ticker: str, config: dict) -> pd.DataFrame:
    """Obtiene datos de acciones/índices via yfinance."""
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
            f"Verifica que el símbolo sea correcto (ej: SPY, AAPL, BTC-USD)."
        )

    # yfinance puede devolver MultiIndex en columnas; aplanar
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Asegurar que las columnas estén en el formato esperado
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

        # Mapeo de timeframe para ccxt
        tf_map = {"30m": "30m", "4H": "1h", "1D": "1d", "1W": "1d"}
        ccxt_tf = tf_map.get(timeframe, "1d")

        # Calcular cuántas velas necesitar (approx)
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
    """Calcula retornos logarítmicos: r_t = ln(C_t / C_{t-1})"""
    return np.log(close / close.shift(1))


def compute_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """Volatilidad realizada como desviación estándar móvil de retornos."""
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

    # Eliminar filas con NaN (las primeras por rolling windows)
    features.dropna(inplace=True)

    if features.empty:
        raise ValueError(
            "No quedan datos después de calcular features. "
            "Los datos son insuficientes para las ventanas móviles configuradas."
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
        # Revisar si hubo warning de convergencia
        for w in caught:
            if "ConvergenceWarning" in str(w.category.__name__):
                converged = False

    return model, converged


def decode_regimes(model: GaussianHMM, features_scaled: np.ndarray) -> np.ndarray:
    """Decodifica la secuencia de regímenes usando el algoritmo de Viterbi."""
    return model.predict(features_scaled)


def label_regimes(model: GaussianHMM, n_regimes: int):
    """
    Ordena regímenes por media de log-return (feature 0) y asigna labels y colores.
    Returns: (label_map dict, color_map dict, sorted_order list)
    """
    # Medias de log-return por régimen
    mean_returns = model.means_[:, 0]

    # Ordenar de menor (bear) a mayor (bull)
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
    """Extrae la matriz de transición como DataFrame con labels semánticos."""
    transmat = model.transmat_

    # Reordenar filas y columnas según sorted_order
    reordered = transmat[sorted_order][:, sorted_order]

    labels = [label_map[idx] for idx in sorted_order]
    return pd.DataFrame(reordered, index=labels, columns=labels)


# ============================================================================
# SECTION E: Visualization
# ============================================================================

def plot_candlestick_with_regimes(
    df: pd.DataFrame,
    regimes: np.ndarray,
    label_map: dict,
    color_map: dict,
    feature_index: pd.Index,
) -> go.Figure:
    """
    Gráfico de velas interactivo con fondo coloreado por régimen detectado.
    El fondo usa vrect semi-transparente para no ocultar las velas.
    """
    # Alinear df con las features (que tienen menos filas por el dropna)
    df_aligned = df.loc[feature_index]

    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_aligned.index,
            open=df_aligned["Open"].squeeze(),
            high=df_aligned["High"].squeeze(),
            low=df_aligned["Low"].squeeze(),
            close=df_aligned["Close"].squeeze(),
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Encontrar bloques contiguos de régimen y dibujar vrects
    regimes_added_to_legend = set()
    i = 0
    while i < len(regimes):
        regime = regimes[i]
        start_idx = i
        while i < len(regimes) and regimes[i] == regime:
            i += 1
        end_idx = i - 1

        label = label_map[regime]
        color = color_map[regime]
        show_legend = regime not in regimes_added_to_legend

        # vrect para el fondo coloreado
        x0 = df_aligned.index[start_idx]
        x1 = df_aligned.index[min(end_idx, len(df_aligned.index) - 1)]

        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=0.15,
            layer="below",
            line_width=0,
        )

        # Scatter invisible para la leyenda
        if show_legend:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=color),
                    name=label,
                    showlegend=True,
                )
            )
            regimes_added_to_legend.add(regime)

    fig.update_layout(
        title="Candlestick con Regímenes HMM",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def plot_transition_heatmap(transition_df: pd.DataFrame) -> go.Figure:
    """Heatmap de la matriz de transición con probabilidades anotadas."""
    labels = transition_df.columns.tolist()
    z = transition_df.values

    # Texto de anotaciones (probabilidades con 2 decimales)
    text = [[f"{val:.2f}" for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="Blues",
            showscale=True,
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        title="Matriz de Transición entre Regímenes",
        xaxis_title="Estado Destino (t+1)",
        yaxis_title="Estado Origen (t)",
        template="plotly_dark",
        height=400,
        yaxis=dict(autorange="reversed"),
    )

    return fig


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
            f"No se pueden ajustar {n_regimes} regímenes a estos datos. "
            f"Intenta con menos regímenes. Detalle: {e}"
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
    )

    st.title("📡 Radar de Tendencia HMM")
    st.caption("Detección de regímenes de mercado con Hidden Markov Models")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuración")

        ticker = st.text_input(
            "Ticker / Símbolo",
            value="SPY",
            help="Acciones: SPY, AAPL, MSFT | Crypto: BTC/USDT, ETH/USDT",
        )

        timeframe = st.selectbox(
            "Timeframe",
            options=list(TIMEFRAME_CONFIG.keys()),
            format_func=lambda x: f"{x} ({TIMEFRAME_CONFIG[x]['label']})",
            index=2,  # Default: 1D
        )

        n_regimes = st.slider(
            "Número de Regímenes",
            min_value=3,
            max_value=7,
            value=3,
            help="Cantidad de estados ocultos del HMM (3=simple, 7=granular)",
        )

        train_button = st.button("🚀 Entrenar / Actualizar Modelo", use_container_width=True)

        st.divider()

        with st.expander("📐 Explicación Matemática"):
            st.markdown(MATH_EXPLANATION)

    # ── Main Panel ───────────────────────────────────────────────────────

    # Inicializar session state
    if "trained" not in st.session_state:
        st.session_state.trained = False

    if train_button:
        _run_pipeline(ticker, timeframe, n_regimes)

    # Mostrar resultados si hay un modelo entrenado
    if st.session_state.trained:
        _display_results()
    else:
        st.info(
            "👈 Configura los parámetros en el panel lateral y presiona "
            "**Entrenar / Actualizar Modelo** para comenzar."
        )


def _run_pipeline(ticker: str, timeframe: str, n_regimes: int):
    """Ejecuta el pipeline completo: fetch → features → train → decode."""
    config = TIMEFRAME_CONFIG[timeframe]

    # Step 1: Fetch data
    with st.spinner(f"Obteniendo datos para {ticker} ({timeframe})..."):
        try:
            df = fetch_data(ticker, timeframe)
        except ValueError as e:
            st.error(f"❌ Error de datos: {e}")
            return
        except Exception as e:
            st.error(f"❌ Error inesperado al obtener datos: {e}")
            return

    # Step 2: Compute features
    with st.spinner("Calculando features (retornos, volatilidad, momentum)..."):
        try:
            features = compute_features(df, config)
            features_scaled, scaler = normalize_features(features)
        except ValueError as e:
            st.error(f"❌ Error en features: {e}")
            return

    # Step 3: Train HMM
    with st.spinner(f"Entrenando HMM con {n_regimes} regímenes..."):
        model, converged, error_msg = safe_train(features_scaled, n_regimes)

        if error_msg:
            st.error(f"❌ {error_msg}")
            return

        if not converged:
            st.warning(
                "⚠️ El modelo no convergió completamente. Los resultados pueden "
                "ser menos confiables. Intenta reducir regímenes o usar un timeframe más largo."
            )

    # Step 4: Decode regimes
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

    st.success(
        f"✅ Modelo entrenado: {len(regimes)} barras analizadas, "
        f"{n_regimes} regímenes detectados."
    )


def _display_results():
    """Muestra los resultados del modelo entrenado."""
    model = st.session_state.model
    df = st.session_state.df
    features = st.session_state.features
    regimes = st.session_state.regimes
    label_map = st.session_state.label_map
    color_map = st.session_state.color_map
    transition_df = st.session_state.transition_df
    ticker = st.session_state.ticker
    timeframe = st.session_state.timeframe

    # ── Régimen actual ───────────────────────────────────────────────
    current_regime = regimes[-1]
    current_label = label_map[current_regime]
    current_color = color_map[current_regime]

    # Panel superior: métricas del régimen actual
    col_regime, col_metrics1, col_metrics2, col_metrics3 = st.columns([2, 1, 1, 1])

    with col_regime:
        st.markdown(
            f"""
            <div style="background-color: {current_color}22; border-left: 5px solid {current_color};
                        padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                <h3 style="margin: 0; color: {current_color};">Régimen Actual: {current_label}</h3>
                <p style="margin: 5px 0 0 0; opacity: 0.8;">{ticker} — {timeframe}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Métricas del régimen actual (valores medios del modelo)
    regime_means = model.means_[current_regime]
    with col_metrics1:
        st.metric("Retorno Medio", f"{regime_means[0]:.4f}")
    with col_metrics2:
        st.metric("Volatilidad Media", f"{regime_means[1]:.4f}")
    with col_metrics3:
        st.metric("Momentum Medio", f"{regime_means[2]:.4f}")

    st.divider()

    # ── Gráfico principal + probabilidades de transición ──────────────
    col_chart, col_probs = st.columns([3, 1])

    with col_chart:
        fig_candle = plot_candlestick_with_regimes(
            df, regimes, label_map, color_map, features.index
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    with col_probs:
        st.subheader("Probabilidades de Transición")
        st.caption(f"Desde: **{current_label}**")

        # Probabilidades desde el régimen actual
        trans_probs = model.transmat_[current_regime]
        for regime_idx in st.session_state.sorted_order:
            prob = trans_probs[regime_idx]
            label = label_map[regime_idx]
            color = color_map[regime_idx]

            # Barra de progreso visual
            st.markdown(
                f'<span style="color: {color}; font-weight: bold;">{label}</span>',
                unsafe_allow_html=True,
            )
            st.progress(float(prob), text=f"{prob:.1%}")

        st.divider()

        # Distribución de regímenes en la muestra
        st.subheader("Distribución")
        for regime_idx in st.session_state.sorted_order:
            count = np.sum(regimes == regime_idx)
            pct = count / len(regimes)
            label = label_map[regime_idx]
            color = color_map[regime_idx]
            st.markdown(
                f'<span style="color: {color};">{label}: {pct:.1%} ({count} barras)</span>',
                unsafe_allow_html=True,
            )

    # ── Matriz de transición (heatmap) ────────────────────────────────
    st.divider()
    st.subheader("Matriz de Transición Completa")

    col_heatmap, col_table = st.columns([2, 1])

    with col_heatmap:
        fig_heatmap = plot_transition_heatmap(transition_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col_table:
        st.caption("Valores numéricos de la matriz:")
        styled_df = transition_df.style.format("{:.3f}").background_gradient(
            cmap="Blues", vmin=0, vmax=1
        )
        st.dataframe(styled_df, use_container_width=True)

    # ── Features plot (extra insight) ─────────────────────────────────
    with st.expander("📊 Features del Modelo (Debug / Análisis)"):
        fig_features = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Log Returns", "Volatilidad Realizada", "Momentum (EMA Slope)"],
            vertical_spacing=0.08,
        )

        fig_features.add_trace(
            go.Scatter(
                x=features.index,
                y=features["log_return"],
                name="Log Return",
                line=dict(color="#42a5f5", width=1),
            ),
            row=1,
            col=1,
        )
        fig_features.add_trace(
            go.Scatter(
                x=features.index,
                y=features["volatility"],
                name="Volatilidad",
                line=dict(color="#ffa726", width=1),
            ),
            row=2,
            col=1,
        )
        fig_features.add_trace(
            go.Scatter(
                x=features.index,
                y=features["momentum"],
                name="Momentum",
                line=dict(color="#ab47bc", width=1),
            ),
            row=3,
            col=1,
        )

        fig_features.update_layout(
            template="plotly_dark",
            height=500,
            showlegend=False,
        )
        st.plotly_chart(fig_features, use_container_width=True)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
