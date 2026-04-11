"""
Walk-Forward Backtest V4 — Directional Engine (Indicators + MTF Cascade)
========================================================================
Motor direccional nuevo basado en indicadores tecnicos clasicos con
cascada multi-timeframe (1W -> 1D -> 4H).

Indicadores (calculados sin pandas-ta, puro numpy/pandas):
  1. Squeeze Momentum (LazyBear): Bollinger dentro/fuera de Keltner + momentum
  2. ADX (14): fuerza del movimiento + DI+/DI- para direccion
  3. EMA 10/55: cruce y distancia como proxy de tendencia
  4. Accumulation/Distribution Line: presion compradora/vendedora
  5. Volume Profile + Point of Control: nivel de precio con max volumen

Cascada MTF: 1W constrains 1D constrains 4H.
Si 1W es fuerte bajista, 1D no puede ser alcista.

Los rangos del Monte Carlo NO se tocan.

Par: BTC/USDT | Temporalidades: 4H, 1D, 1W
"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from app import (
    TIMEFRAME_CONFIG,
    fetch_data,
    resample_ohlcv,
    compute_features,
    normalize_features,
    train_hmm,
    decode_regimes,
    label_regimes,
    compute_range_projection,
)

# ============================================================================
# CONFIG
# ============================================================================

TICKER = "BTC/USDT"
TIMEFRAMES_ORDER = ["1W", "1D", "4H"]   # cascada: mayor a menor
N_REGIMES = 3
N_WINDOWS = 15      # mas ventanas para significancia
HORIZON = 10
Z_SCORE = 2.0
N_SIMULATIONS = 10000

# Pesos de fusion por timeframe
WEIGHT_PROFILES = {
    "4H":  {"squeeze": 0.30, "adx": 0.20, "ema": 0.20, "ad": 0.15, "vpoc": 0.15},
    "1D":  {"squeeze": 0.25, "adx": 0.25, "ema": 0.20, "ad": 0.15, "vpoc": 0.15},
    "1W":  {"squeeze": 0.25, "adx": 0.25, "ema": 0.20, "ad": 0.15, "vpoc": 0.15},
}

# Fuerza de la restriccion del TF superior sobre el inferior
# 0 = sin restriccion, 1 = TF superior domina completamente
MTF_CONSTRAINT_STRENGTH = 0.40


# ============================================================================
# INDICATOR 1: Squeeze Momentum (LazyBear)
# ============================================================================

def compute_squeeze_momentum(close: pd.Series, high: pd.Series, low: pd.Series,
                              bb_length: int = 20, bb_mult: float = 2.0,
                              kc_length: int = 20, kc_mult: float = 1.5,
                              mom_length: int = 12) -> dict:
    """
    Squeeze Momentum Indicator (LazyBear):
    - Squeeze ON: Bollinger Bands DENTRO de Keltner Channels (vol comprimida)
    - Squeeze OFF: BB FUERA de KC (vol expandiendo)
    - Momentum: regresion lineal del delta (close - midline) sobre mom_length

    Cuando el squeeze se libera, el momentum indica la direccion del breakout.

    Returns:
    - squeeze_on: bool (True = vol comprimida, preparando movimiento)
    - momentum: float (positivo = bullish, negativo = bearish)
    - mom_increasing: bool (momentum acelerando en su direccion)
    - score: [-1, 1] senal direccional
    """
    if len(close) < max(bb_length, kc_length, mom_length) + 10:
        return {"score": 0.0, "squeeze_on": False, "momentum": 0.0,
                "mom_color": "gray", "confidence": 0.0}

    # -- Bollinger Bands --
    bb_mid = close.rolling(bb_length).mean()
    bb_std = close.rolling(bb_length).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # -- Keltner Channels (usando ATR) --
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_length).mean()
    kc_mid = close.rolling(kc_length).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    # -- Squeeze detection --
    squeeze_on = bool(bb_upper.iloc[-1] < kc_upper.iloc[-1] and
                      bb_lower.iloc[-1] > kc_lower.iloc[-1])

    # -- Momentum: linear regression value --
    # delta = close - avg(highest_high, lowest_low, sma)
    highest = high.rolling(kc_length).max()
    lowest = low.rolling(kc_length).min()
    midline = (highest + lowest + kc_mid) / 3
    delta = close - midline

    # Regresion lineal sobre los ultimos mom_length valores de delta
    delta_recent = delta.iloc[-mom_length:].dropna()
    if len(delta_recent) < mom_length:
        return {"score": 0.0, "squeeze_on": squeeze_on, "momentum": 0.0,
                "mom_color": "gray", "confidence": 0.0}

    x = np.arange(len(delta_recent), dtype=float)
    y = delta_recent.values.astype(float)
    # linregress value = ultimo valor de la regresion
    coeffs = np.polyfit(x, y, 1)
    momentum_val = float(coeffs[0] * (len(x) - 1) + coeffs[1])  # valor en x[-1]

    # Momentum anterior (para detectar aceleracion)
    if len(delta) >= mom_length + 1:
        delta_prev = delta.iloc[-(mom_length+1):-1].dropna()
        if len(delta_prev) >= mom_length:
            x_prev = np.arange(len(delta_prev), dtype=float)
            y_prev = delta_prev.values.astype(float)
            c_prev = np.polyfit(x_prev, y_prev, 1)
            momentum_prev = float(c_prev[0] * (len(x_prev) - 1) + c_prev[1])
        else:
            momentum_prev = momentum_val
    else:
        momentum_prev = momentum_val

    # Color del histograma LazyBear:
    # lime = positivo y creciendo, green = positivo y decreciendo
    # red = negativo y decreciendo, maroon = negativo y creciendo
    if momentum_val > 0:
        if momentum_val > momentum_prev:
            mom_color = "lime"       # bull acelerando
        else:
            mom_color = "darkgreen"  # bull desacelerando
    else:
        if momentum_val < momentum_prev:
            mom_color = "red"        # bear acelerando
        else:
            mom_color = "maroon"     # bear desacelerando

    # Normalizar momentum por precio para score
    mom_normalized = momentum_val / float(close.iloc[-1]) * 100

    # Score
    if squeeze_on:
        # En squeeze: el momentum indica la direccion del breakout inminente
        # pero con confianza moderada (aun no confirmo)
        score = float(np.clip(mom_normalized * 5, -0.7, 0.7))
        confidence = 0.5 + abs(score) * 0.3
    else:
        # Squeeze released: alta confianza en la direccion
        score = float(np.clip(mom_normalized * 8, -1.0, 1.0))
        # Bonus si el momentum esta acelerando
        if mom_color in ("lime", "red"):
            score *= 1.2
        confidence = min(abs(score), 1.0)

    score = float(np.clip(score, -1, 1))

    return {
        "score": round(score, 3),
        "squeeze_on": squeeze_on,
        "momentum": round(momentum_val, 4),
        "mom_normalized": round(mom_normalized, 4),
        "mom_color": mom_color,
        "confidence": round(min(confidence, 1.0), 3),
    }


# ============================================================================
# INDICATOR 2: ADX (Average Directional Index)
# ============================================================================

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> dict:
    """
    ADX: mide FUERZA de la tendencia (no direccion directamente).
    +DI y -DI dan la direccion.

    ADX > 25 = tendencia fuerte -> confiar en la direccion de DI
    ADX < 20 = sin tendencia -> no hay edge direccional
    ADX subiendo = tendencia fortaleciendo

    Returns:
    - adx: float [0, 100]
    - plus_di: float
    - minus_di: float
    - trending: bool (ADX > 25)
    - score: [-1, 1] basado en DI ponderado por ADX
    """
    if len(close) < period * 3:
        return {"score": 0.0, "adx": 0.0, "plus_di": 0.0, "minus_di": 0.0,
                "trending": False, "confidence": 0.0}

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smoothed averages (Wilder's smoothing = EMA with alpha=1/period)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)

    # DX and ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    current_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
    current_plus = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0
    current_minus = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0

    trending = current_adx > 25

    # ADX trend (subiendo o bajando en ultimos 5 periodos)
    adx_recent = adx.iloc[-5:].dropna()
    adx_rising = False
    if len(adx_recent) >= 3:
        adx_rising = float(adx_recent.iloc[-1]) > float(adx_recent.iloc[0])

    # Score: direccion de DI, ponderada por fuerza del ADX
    di_score = (current_plus - current_minus) / max(current_plus + current_minus, 1)

    if trending:
        # Tendencia fuerte: confiar en DI
        score = di_score * min(current_adx / 40, 1.0)  # scale by ADX strength
        if adx_rising:
            score *= 1.15  # bonus si tendencia fortaleciendo
        confidence = min(current_adx / 50, 1.0)
    else:
        # Sin tendencia: senal debil
        score = di_score * 0.3
        confidence = 0.2

    score = float(np.clip(score, -1, 1))

    return {
        "score": round(score, 3),
        "adx": round(current_adx, 1),
        "plus_di": round(current_plus, 1),
        "minus_di": round(current_minus, 1),
        "trending": trending,
        "adx_rising": adx_rising,
        "confidence": round(confidence, 3),
    }


# ============================================================================
# INDICATOR 3: EMA 10/55 Cross
# ============================================================================

def compute_ema_signal(close: pd.Series, fast: int = 10, slow: int = 55) -> dict:
    """
    EMA 10/55:
    - EMA10 > EMA55 = alcista
    - EMA10 < EMA55 = bajista
    - Distancia entre ellas = fuerza de la tendencia
    - Cruce reciente = cambio de tendencia
    - Pendiente de ambas EMAs = aceleracion

    Returns score [-1, 1]
    """
    if len(close) < slow + 10:
        return {"score": 0.0, "cross": "NEUTRAL", "distance_pct": 0.0,
                "confidence": 0.0}

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    current_fast = float(ema_fast.iloc[-1])
    current_slow = float(ema_slow.iloc[-1])
    current_price = float(close.iloc[-1])

    # Distancia normalizada
    distance_pct = (current_fast - current_slow) / current_slow * 100

    # Cross detection (en ultimas 5 barras)
    recent_diff = (ema_fast - ema_slow).iloc[-5:]
    cross_type = "NONE"
    if len(recent_diff) >= 2:
        for i in range(1, len(recent_diff)):
            if recent_diff.iloc[i-1] < 0 and recent_diff.iloc[i] > 0:
                cross_type = "GOLDEN"   # bullish cross
            elif recent_diff.iloc[i-1] > 0 and recent_diff.iloc[i] < 0:
                cross_type = "DEATH"    # bearish cross

    # Pendientes de ambas EMAs (ultimos 5 periodos)
    fast_slope = float(ema_fast.iloc[-1] - ema_fast.iloc[-5]) / current_price * 100
    slow_slope = float(ema_slow.iloc[-1] - ema_slow.iloc[-5]) / current_price * 100

    # Precio respecto a ambas EMAs
    price_vs_fast = (current_price - current_fast) / current_fast * 100
    price_vs_slow = (current_price - current_slow) / current_slow * 100

    # Score compuesto
    # 40% posicion relativa EMA10 vs EMA55
    pos_score = float(np.clip(distance_pct / 3, -1, 1))

    # 30% pendiente de EMA rapida (aceleracion)
    slope_score = float(np.clip(fast_slope * 5, -1, 1))

    # 20% precio vs EMA lenta (soporte/resistencia)
    pvs_score = float(np.clip(price_vs_slow / 5, -1, 1))

    # 10% bonus cruce reciente
    cross_score = 0.0
    if cross_type == "GOLDEN":
        cross_score = 0.8
    elif cross_type == "DEATH":
        cross_score = -0.8

    score = 0.40 * pos_score + 0.30 * slope_score + 0.20 * pvs_score + 0.10 * cross_score
    score = float(np.clip(score, -1, 1))

    # Cross label para display
    if current_fast > current_slow:
        cross = f"BULL ({distance_pct:+.1f}%)"
    else:
        cross = f"BEAR ({distance_pct:+.1f}%)"

    confidence = min(abs(score) * 1.3, 1.0)

    return {
        "score": round(score, 3),
        "cross": cross,
        "distance_pct": round(distance_pct, 2),
        "fast_slope": round(fast_slope, 3),
        "slow_slope": round(slow_slope, 3),
        "cross_type": cross_type,
        "confidence": round(confidence, 3),
    }


# ============================================================================
# INDICATOR 4: Accumulation/Distribution Line
# ============================================================================

def compute_ad_line(high: pd.Series, low: pd.Series, close: pd.Series,
                    volume: pd.Series, lookback: int = 20) -> dict:
    """
    Accumulation/Distribution Line:
    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    AD = cumsum(CLV * Volume)

    Divergencia AD vs Precio:
    - AD sube + precio baja = acumulacion (smart money comprando) -> UP
    - AD baja + precio sube = distribucion (smart money vendiendo) -> DOWN

    Returns score [-1, 1]
    """
    if len(close) < lookback + 5:
        return {"score": 0.0, "ad_slope": 0.0, "divergence": "NONE",
                "confidence": 0.0}

    # Close Location Value
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    clv = clv.fillna(0)

    # AD Line
    ad = (clv * volume).cumsum()

    # Slopes normalizados (ultimos lookback periodos)
    ad_recent = ad.iloc[-lookback:]
    close_recent = close.iloc[-lookback:]

    # Normalizar para comparar
    def norm_slope(series, window=10):
        s = series.iloc[-window:]
        if len(s) < 3:
            return 0.0
        x = np.arange(len(s), dtype=float)
        y = (s - s.iloc[0]).values.astype(float)
        y_range = max(abs(y.max()), abs(y.min()), 1e-10)
        y_norm = y / y_range
        return float(np.polyfit(x, y_norm, 1)[0])

    ad_slope = norm_slope(ad_recent)
    price_slope = norm_slope(close_recent)

    # Detectar divergencia
    threshold = 0.02
    if ad_slope > threshold and price_slope < -threshold:
        divergence = "BULL_DIV"      # acumulacion: AD sube, precio baja
        score = 0.7 + min(abs(ad_slope - price_slope) * 3, 0.3)
    elif ad_slope < -threshold and price_slope > threshold:
        divergence = "BEAR_DIV"      # distribucion: AD baja, precio sube
        score = -(0.7 + min(abs(ad_slope - price_slope) * 3, 0.3))
    elif ad_slope > threshold and price_slope > threshold:
        divergence = "CONFIRM_UP"    # ambos suben = confirmacion alcista
        score = 0.4
    elif ad_slope < -threshold and price_slope < -threshold:
        divergence = "CONFIRM_DOWN"  # ambos bajan = confirmacion bajista
        score = -0.4
    else:
        divergence = "NEUTRAL"
        score = float(np.clip(ad_slope * 5, -0.3, 0.3))

    score = float(np.clip(score, -1, 1))

    return {
        "score": round(score, 3),
        "ad_slope": round(ad_slope, 4),
        "price_slope": round(price_slope, 4),
        "divergence": divergence,
        "confidence": round(min(abs(score), 1.0), 3),
    }


# ============================================================================
# INDICATOR 5: Volume Profile + Point of Control
# ============================================================================

def compute_volume_profile(close: pd.Series, volume: pd.Series,
                            high: pd.Series, low: pd.Series,
                            lookback: int = 40, n_bins: int = 20) -> dict:
    """
    Volume Profile: distribucion del volumen por nivel de precio.
    Point of Control (POC): nivel con maximo volumen.

    - Precio > POC = zona de demanda debajo, soporte -> bullish
    - Precio < POC = zona de oferta arriba, resistencia -> bearish
    - Value Area (70% del volumen): define el rango operativo

    Returns score [-1, 1]
    """
    if len(close) < lookback:
        return {"score": 0.0, "poc": 0.0, "price_vs_poc": 0.0,
                "va_high": 0.0, "va_low": 0.0, "confidence": 0.0}

    c = close.iloc[-lookback:]
    v = volume.iloc[-lookback:]
    h = high.iloc[-lookback:]
    l = low.iloc[-lookback:]

    price_min = float(l.min())
    price_max = float(h.max())

    if price_max == price_min:
        return {"score": 0.0, "poc": float(c.iloc[-1]), "price_vs_poc": 0.0,
                "va_high": price_max, "va_low": price_min, "confidence": 0.0}

    # Crear bins de precio
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_volume = np.zeros(n_bins)

    # Distribuir volumen de cada barra en los bins que toca
    for i in range(len(c)):
        bar_low = float(l.iloc[i])
        bar_high = float(h.iloc[i])
        bar_vol = float(v.iloc[i])

        if bar_high == bar_low:
            # Barra doji: todo el volumen en un bin
            bin_idx = min(int((bar_low - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
            bin_volume[bin_idx] += bar_vol
        else:
            # Distribuir volumen proporcionalmente
            for j in range(n_bins):
                bin_low = bin_edges[j]
                bin_high = bin_edges[j + 1]
                overlap = max(0, min(bar_high, bin_high) - max(bar_low, bin_low))
                bar_range = bar_high - bar_low
                if bar_range > 0:
                    bin_volume[j] += bar_vol * (overlap / bar_range)

    # Point of Control: bin con mayor volumen
    poc_bin = int(np.argmax(bin_volume))
    poc = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2)

    # Value Area: bins que contienen el 70% del volumen total
    total_vol = bin_volume.sum()
    if total_vol == 0:
        return {"score": 0.0, "poc": poc, "price_vs_poc": 0.0,
                "va_high": price_max, "va_low": price_min, "confidence": 0.0}

    # Expandir desde POC hasta cubrir 70%
    sorted_bins = np.argsort(bin_volume)[::-1]
    cumvol = 0.0
    va_bins = set()
    for idx in sorted_bins:
        va_bins.add(idx)
        cumvol += bin_volume[idx]
        if cumvol >= 0.70 * total_vol:
            break

    va_low = float(bin_edges[min(va_bins)])
    va_high = float(bin_edges[max(va_bins) + 1])

    current_price = float(c.iloc[-1])

    # Score basado en posicion del precio respecto al POC y Value Area
    price_vs_poc = (current_price - poc) / poc * 100

    if current_price > va_high:
        # Precio por encima del Value Area = breakout alcista o sobreextencion
        score = 0.3  # moderadamente bullish (soporte debajo)
    elif current_price < va_low:
        # Precio por debajo del Value Area = breakout bajista
        score = -0.3
    elif current_price > poc:
        # Dentro del VA, arriba del POC = bullish
        relative_pos = (current_price - poc) / (va_high - poc) if va_high > poc else 0
        score = 0.2 + 0.4 * relative_pos
    else:
        # Dentro del VA, abajo del POC = bearish
        relative_pos = (poc - current_price) / (poc - va_low) if poc > va_low else 0
        score = -(0.2 + 0.4 * relative_pos)

    score = float(np.clip(score, -1, 1))
    confidence = min(abs(score) * 1.5, 1.0)

    return {
        "score": round(score, 3),
        "poc": round(poc, 2),
        "price_vs_poc": round(price_vs_poc, 2),
        "va_high": round(va_high, 2),
        "va_low": round(va_low, 2),
        "confidence": round(confidence, 3),
    }


# ============================================================================
# DIRECTIONAL FUSION + MTF CASCADE
# ============================================================================

def compute_directional_score(close: pd.Series, high: pd.Series,
                               low: pd.Series, volume: pd.Series,
                               timeframe: str) -> dict:
    """
    Calcula el score direccional combinando los 5 indicadores.
    Returns score [-1, 1] y todos los componentes.
    """
    weights = WEIGHT_PROFILES.get(timeframe, WEIGHT_PROFILES["1D"])

    # Calcular indicadores
    squeeze = compute_squeeze_momentum(close, high, low)
    adx = compute_adx(high, low, close)
    ema = compute_ema_signal(close)
    ad = compute_ad_line(high, low, close, volume)
    vpoc = compute_volume_profile(close, volume, high, low)

    # Fusion ponderada con ajuste por confianza
    components = {
        "squeeze": (squeeze["score"], squeeze["confidence"], weights["squeeze"]),
        "adx": (adx["score"], adx["confidence"], weights["adx"]),
        "ema": (ema["score"], ema["confidence"], weights["ema"]),
        "ad": (ad["score"], ad["confidence"], weights["ad"]),
        "vpoc": (vpoc["score"], vpoc["confidence"], weights["vpoc"]),
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for name, (score, conf, weight) in components.items():
        adj_weight = weight * (0.3 + 0.7 * conf)
        weighted_sum += adj_weight * score
        weight_total += adj_weight

    if weight_total > 0:
        fused_score = weighted_sum / weight_total
    else:
        fused_score = 0.0

    fused_score = float(np.clip(fused_score, -1, 1))

    # Agreement check
    scores = [squeeze["score"], adx["score"], ema["score"], ad["score"], vpoc["score"]]
    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)

    if pos >= 4:
        agreement = "STRONG UP"
    elif neg >= 4:
        agreement = "STRONG DOWN"
    elif pos >= 3:
        agreement = "LEAN UP"
    elif neg >= 3:
        agreement = "LEAN DOWN"
    else:
        agreement = "MIXED"

    prob_up = 50 + fused_score * 50
    prob_up = float(np.clip(prob_up, 1, 99))

    return {
        "score": round(fused_score, 3),
        "prob_up": round(prob_up, 1),
        "direction": "UP" if fused_score > 0 else "DOWN",
        "agreement": agreement,
        # Components
        "sqz_s": round(squeeze["score"], 3),
        "sqz_on": squeeze["squeeze_on"],
        "sqz_color": squeeze["mom_color"],
        "adx_s": round(adx["score"], 3),
        "adx_val": adx["adx"],
        "adx_trend": adx["trending"],
        "ema_s": round(ema["score"], 3),
        "ema_cross": ema["cross"],
        "ad_s": round(ad["score"], 3),
        "ad_div": ad["divergence"],
        "vpoc_s": round(vpoc["score"], 3),
        "vpoc_vs": vpoc["price_vs_poc"],
    }


def apply_mtf_constraint(lower_tf_score: float, higher_tf_result: dict,
                          strength: float = MTF_CONSTRAINT_STRENGTH) -> float:
    """
    Aplica la restriccion del timeframe superior sobre el inferior.

    Si 1W es fuerte bajista (score < -0.3 con agreement STRONG/LEAN DOWN),
    el 1D no puede ser alcista — se sesga hacia abajo.

    La restriccion es proporcional a la fuerza y al agreement del TF superior.
    """
    if higher_tf_result is None:
        return lower_tf_score

    htf_score = higher_tf_result["score"]
    htf_agreement = higher_tf_result["agreement"]

    # Fuerza de la restriccion basada en el agreement
    if "STRONG" in htf_agreement:
        constraint_factor = strength * 1.2
    elif "LEAN" in htf_agreement:
        constraint_factor = strength * 0.8
    else:
        constraint_factor = strength * 0.3  # MIXED = restriccion debil

    # Si los TFs van en la misma direccion: reforzar
    # Si van en direcciones opuestas: el superior restringe al inferior
    if np.sign(lower_tf_score) == np.sign(htf_score):
        # Misma direccion: boost moderado
        constrained = lower_tf_score + htf_score * constraint_factor * 0.3
    else:
        # Direcciones opuestas: el HTF tira al LTF hacia el
        constrained = lower_tf_score + htf_score * constraint_factor

    return float(np.clip(constrained, -1, 1))


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_single_window(features_all, close_all, df_all, cutoff_idx,
                      horizon, n_regimes, z_score, config, timeframe,
                      htf_result=None):
    """Entrena HMM + calcula direction V4 con constraint del TF superior."""

    features_train = features_all.iloc[:cutoff_idx]
    close_train = close_all.iloc[:cutoff_idx]

    test_end = min(cutoff_idx + horizon, len(close_all))
    close_test = close_all.iloc[cutoff_idx:test_end]
    actual_horizon = len(close_test)

    if actual_horizon == 0:
        return None

    # -- Train HMM --
    scaled_train, scaler = normalize_features(features_train)
    model, converged = train_hmm(scaled_train, n_regimes)
    regimes = decode_regimes(model, scaled_train)
    current_regime = regimes[-1]
    label_map, _, _ = label_regimes(model, n_regimes)
    current_label = label_map[current_regime]
    last_close = float(close_train.iloc[-1])

    # -- Monte Carlo (rangos) --
    range_data = compute_range_projection(
        model, current_regime, last_close,
        actual_horizon, z_score, scaler, N_SIMULATIONS
    )

    # -- OHLCV del train --
    train_end_loc = df_all.index.get_loc(close_train.index[-1])
    df_train = df_all.iloc[:train_end_loc + 1]

    # -- V4 Direction --
    dir_result = compute_directional_score(
        df_train["Close"].squeeze(),
        df_train["High"].squeeze(),
        df_train["Low"].squeeze(),
        df_train["Volume"].squeeze(),
        timeframe,
    )

    # -- Apply MTF constraint --
    raw_score = dir_result["score"]
    constrained_score = apply_mtf_constraint(raw_score, htf_result)

    # Recalcular direction con constraint
    v4_prob_up = 50 + constrained_score * 50
    v4_prob_up = float(np.clip(v4_prob_up, 1, 99))
    v4_dir = "UP" if constrained_score > 0 else "DOWN"

    # -- Evaluar --
    real_prices = close_test.values
    real_final = float(real_prices[-1])
    actual_dir = "UP" if real_final > last_close else "DOWN"

    v1_dir = "UP" if range_data["prob_up"] > 50 else "DOWN"
    v1_correct = v1_dir == actual_dir
    v4_correct = v4_dir == actual_dir

    # Rangos
    inside_iqr = 0
    inside_full = 0
    for i in range(actual_horizon):
        p = float(real_prices[i])
        if range_data["tunnel_p25"][i] <= p <= range_data["tunnel_p75"][i]:
            inside_iqr += 1
        if range_data["tunnel_lower"][i] <= p <= range_data["tunnel_upper"][i]:
            inside_full += 1

    real_return_pct = (real_final - last_close) / last_close * 100

    return {
        "cutoff_date": features_train.index[-1],
        "regime": current_label,
        "real_return_pct": round(real_return_pct, 2),
        # V1
        "v1_prob_up": round(range_data["prob_up"], 1),
        "v1_dir": v1_dir,
        "v1_correct": v1_correct,
        # V4
        "v4_prob_up": round(v4_prob_up, 1),
        "v4_dir": v4_dir,
        "v4_correct": v4_correct,
        "v4_raw": round(raw_score, 3),
        "v4_constrained": round(constrained_score, 3),
        "agreement": dir_result["agreement"],
        # Components
        "sqz_s": dir_result["sqz_s"],
        "sqz_on": dir_result["sqz_on"],
        "sqz_clr": dir_result["sqz_color"][:1].upper(),
        "adx_s": dir_result["adx_s"],
        "adx_v": dir_result["adx_val"],
        "ema_s": dir_result["ema_s"],
        "ad_s": dir_result["ad_s"],
        "ad_div": dir_result["ad_div"],
        "vpoc_s": dir_result["vpoc_s"],
        # Ranges
        "pct_inside_iqr": round(inside_iqr / actual_horizon * 100, 1),
        "pct_inside_full": round(inside_full / actual_horizon * 100, 1),
        # Para cascada
        "_dir_result": dir_result,
    }


def prepare_timeframe_data(ticker, timeframe):
    """Fetch y prepara datos para un timeframe."""
    config = TIMEFRAME_CONFIG[timeframe]
    df = fetch_data(ticker, timeframe)
    if config["resample_rule"]:
        df = resample_ohlcv(df, config["resample_rule"])
    features = compute_features(df, config)
    close = df["Close"].squeeze().loc[features.index]
    return df, features, close, config


def run_backtest_cascade(ticker, n_regimes, n_windows, horizon, z_score):
    """
    Corre el backtest con cascada MTF: 1W -> 1D -> 4H.
    Para cada ventana temporal, primero corre 1W, usa su resultado
    para constrainar 1D, y el de 1D para constrainar 4H.
    """
    # Pre-cargar todos los datos
    print(f"\n  Cargando datos...")
    tf_data = {}
    for tf in TIMEFRAMES_ORDER:
        df, features, close, config = prepare_timeframe_data(ticker, tf)
        tf_data[tf] = {"df": df, "features": features, "close": close, "config": config}
        print(f"  {tf}: {len(df)} barras | {len(features)} features | "
              f"{df.index[0]} -> {df.index[-1]}")

    all_summaries = {}

    # --- Correr cada timeframe en orden (mayor a menor) ---
    for tf_idx, tf in enumerate(TIMEFRAMES_ORDER):
        print(f"\n{'='*95}")
        weights = WEIGHT_PROFILES.get(tf, WEIGHT_PROFILES["1D"])
        print(f"  {tf} | Squeeze={weights['squeeze']} ADX={weights['adx']} "
              f"EMA={weights['ema']} A/D={weights['ad']} VPOC={weights['vpoc']}")
        if tf_idx > 0:
            htf = TIMEFRAMES_ORDER[tf_idx - 1]
            print(f"  Constrained by: {htf} (strength={MTF_CONSTRAINT_STRENGTH})")
        print(f"{'='*95}")

        data = tf_data[tf]
        features_all = data["features"]
        close_all = data["close"]
        df_all = data["df"]
        config = data["config"]

        actual_windows = min(n_windows, (len(features_all) - config["min_bars_required"]) // horizon)
        if actual_windows < 1:
            print(f"  SKIP {tf}: datos insuficientes")
            continue

        print(f"  Ventanas: {actual_windows} x {horizon} periodos\n")

        # Header
        print(f"  {'W':>2s} | {'Corte':>16s} | {'Reg':>8s} | {'Real':>7s} | "
              f"{'V1':>3s} | {'V4':>3s} {'pUp':>5s} | {'Agree':>10s} | "
              f"{'Sqz':>5s} | {'ADX':>5s} | {'EMA':>5s} | {'A/D':>5s} | {'VPOC':>5s} | "
              f"{'IQR':>5s} | {'Full':>5s}")
        print(f"  {'-'*130}")

        results = []

        for w in range(actual_windows):
            offset = (w + 1) * horizon
            cutoff_idx = len(features_all) - offset

            if cutoff_idx < config["min_bars_required"]:
                continue

            # Para la cascada: obtener el htf_result de la misma ventana
            # del timeframe superior
            htf_result = None
            if tf_idx > 0:
                htf = TIMEFRAMES_ORDER[tf_idx - 1]
                if htf in all_summaries and w < len(all_summaries[htf]):
                    htf_result = all_summaries[htf][w].get("_dir_result")

            result = run_single_window(
                features_all, close_all, df_all, cutoff_idx,
                horizon, n_regimes, z_score, config, tf,
                htf_result=htf_result,
            )

            if result is None:
                continue

            results.append(result)

            v1i = "OK" if result["v1_correct"] else "xx"
            v4i = "OK" if result["v4_correct"] else "xx"
            sqz_tag = f"{'S' if result['sqz_on'] else ' '}{result['sqz_clr']}"

            print(f"  [{w+1:>2d}] | {str(result['cutoff_date'])[:16]:>16s} | "
                  f"{result['regime']:>8s} | {result['real_return_pct']:>+6.2f}% | "
                  f"{v1i:>3s} | {v4i:>3s} {result['v4_prob_up']:>4.1f}% | "
                  f"{result['agreement']:>10s} | "
                  f"{result['sqz_s']:>+5.2f} | {result['adx_s']:>+5.2f} | "
                  f"{result['ema_s']:>+5.2f} | {result['ad_s']:>+5.2f} | "
                  f"{result['vpoc_s']:>+5.2f} | "
                  f"{result['pct_inside_iqr']:>4.1f}% | {result['pct_inside_full']:>4.1f}%")

        all_summaries[tf] = results

        if not results:
            continue

        df_r = pd.DataFrame(results)
        n = len(df_r)
        v1_acc = df_r["v1_correct"].sum() / n * 100
        v4_acc = df_r["v4_correct"].sum() / n * 100
        delta = v4_acc - v1_acc
        d_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"

        print(f"\n  {'-'*60}")
        print(f"  RESUMEN {tf} ({n} ventanas)")
        print(f"  {'-'*60}")
        print(f"  V1 Dir (MC only):         {v1_acc:>6.1f}%")
        print(f"  V4 Dir (indicators+MTF):  {v4_acc:>6.1f}%  ({d_str})")
        print(f"  IQR containment:          {df_r['pct_inside_iqr'].mean():>6.1f}%  (~50%)")
        print(f"  Full containment:         {df_r['pct_inside_full'].mean():>6.1f}%  (~95%)")
        print(f"  {'-'*60}")
        print(f"  Signals avg: Sqz={df_r['sqz_s'].mean():>+.3f} | "
              f"ADX={df_r['adx_s'].mean():>+.3f} | EMA={df_r['ema_s'].mean():>+.3f} | "
              f"A/D={df_r['ad_s'].mean():>+.3f} | VPOC={df_r['vpoc_s'].mean():>+.3f}")

    # --- Tabla final ---
    print(f"\n\n{'='*95}")
    print("  COMPARATIVA FINAL -- V1 (MC) vs V4 (Indicators + MTF Cascade)")
    print(f"{'='*95}")
    print(f"  {'TF':>4s} | {'Win':>3s} | {'V1 Dir':>7s} | {'V4 Dir':>7s} | "
          f"{'Delta':>6s} | {'IQR':>6s} | {'Full':>6s}")
    print(f"  {'-'*4} | {'-'*3} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*6}")

    for tf in TIMEFRAMES_ORDER:
        if tf not in all_summaries or not all_summaries[tf]:
            continue
        df_r = pd.DataFrame(all_summaries[tf])
        n = len(df_r)
        v1 = df_r["v1_correct"].sum() / n * 100
        v4 = df_r["v4_correct"].sum() / n * 100
        d = v4 - v1
        ds = f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"
        print(f"  {tf:>4s} | {n:>3d} | {v1:>6.1f}% | {v4:>6.1f}% | "
              f"{ds:>6s} | {df_r['pct_inside_iqr'].mean():>5.1f}% | "
              f"{df_r['pct_inside_full'].mean():>5.1f}%")

    print(f"\n  MTF Cascade: {' -> '.join(TIMEFRAMES_ORDER)}")
    print(f"  Constraint strength: {MTF_CONSTRAINT_STRENGTH}")
    print(f"\n  Backtest V4 finalizado.\n")


def main():
    print("\n" + "=" * 95)
    print("  WALK-FORWARD BACKTEST V4 -- Indicators + MTF Cascade")
    print(f"  Par: {TICKER} | Regimenes: {N_REGIMES} | Horizonte: {HORIZON} | Ventanas: {N_WINDOWS}")
    print(f"  Indicators: Squeeze Momentum | ADX | EMA 10/55 | A/D Line | Volume Profile")
    print(f"  Cascade: 1W -> 1D -> 4H (constraint={MTF_CONSTRAINT_STRENGTH})")
    print("=" * 95)

    run_backtest_cascade(TICKER, N_REGIMES, N_WINDOWS, HORIZON, Z_SCORE)


if __name__ == "__main__":
    main()
