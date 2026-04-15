"""
Walk-Forward Backtest V4b — Hard MTF Cascade + Divergence Detection
====================================================================
Mejora sobre V4:
  A) Cascada DURA: el TF superior impone techo/piso al inferior.
     Si 1W dice DOWN fuerte, 1D NO puede ser alcista (cap en prob_up=40%).
  B) Divergencia de momentum: detecta cuando precio hace nuevo high/low
     pero el Squeeze momentum no confirma = agotamiento = reversal.
  C) Indicadores del 1W calculados sobre data diaria resampleada para
     tener mas granularidad en el calculo.

Par: BTC/USDT | Temporalidades: 1W -> 1D -> 4H
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
TIMEFRAMES_ORDER = ["1W", "1D", "4H"]
N_REGIMES = 3
N_WINDOWS = 15
HORIZON = 10
Z_SCORE = 2.0
N_SIMULATIONS = 10000

# Pesos: mas peso al Squeeze (incluye divergencia) y A/D
WEIGHT_PROFILES = {
    "4H":  {"squeeze": 0.30, "adx": 0.20, "ema": 0.20, "ad": 0.15, "vpoc": 0.15},
    "1D":  {"squeeze": 0.30, "adx": 0.20, "ema": 0.20, "ad": 0.15, "vpoc": 0.15},
    "1W":  {"squeeze": 0.30, "adx": 0.20, "ema": 0.20, "ad": 0.15, "vpoc": 0.15},
}

# Cascada dura: caps de prob_up segun el TF superior
# Si HTF agreement es STRONG DOWN -> LTF prob_up <= 35%
# Si HTF agreement es LEAN DOWN   -> LTF prob_up <= 45%
# Si HTF agreement es STRONG UP   -> LTF prob_up >= 65%
# Si HTF agreement es LEAN UP     -> LTF prob_up >= 55%
MTF_CAPS = {
    "STRONG DOWN": {"max_prob_up": 35},
    "LEAN DOWN":   {"max_prob_up": 45},
    "STRONG UP":   {"min_prob_up": 65},
    "LEAN UP":     {"min_prob_up": 55},
    "MIXED":       {},  # sin restriccion
}


# ============================================================================
# INDICATOR 1: Squeeze Momentum + DIVERGENCE
# ============================================================================

def compute_squeeze_momentum(close, high, low,
                              bb_length=20, bb_mult=2.0,
                              kc_length=20, kc_mult=1.5,
                              mom_length=12) -> dict:
    """
    Squeeze Momentum (LazyBear) + deteccion de divergencia precio-momentum.
    """
    if len(close) < max(bb_length, kc_length, mom_length) + 20:
        return {"score": 0.0, "squeeze_on": False, "momentum": 0.0,
                "mom_color": "gray", "divergence": "NONE", "confidence": 0.0}

    # -- Bollinger Bands --
    bb_mid = close.rolling(bb_length).mean()
    bb_std = close.rolling(bb_length).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # -- Keltner Channels --
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_length).mean()
    kc_mid = close.rolling(kc_length).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    squeeze_on = bool(bb_upper.iloc[-1] < kc_upper.iloc[-1] and
                      bb_lower.iloc[-1] > kc_lower.iloc[-1])

    # -- Momentum via linear regression --
    highest = high.rolling(kc_length).max()
    lowest = low.rolling(kc_length).min()
    midline = (highest + lowest + kc_mid) / 3
    delta = close - midline

    def linreg_value(series, length):
        s = series.iloc[-length:].dropna()
        if len(s) < length:
            return 0.0
        x = np.arange(len(s), dtype=float)
        y = s.values.astype(float)
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0] * (len(x) - 1) + coeffs[1])

    momentum_val = linreg_value(delta, mom_length)

    # Momentum anterior
    if len(delta) >= mom_length + 1:
        delta_shifted = delta.iloc[:-(1)]
        momentum_prev = linreg_value(delta_shifted, mom_length)
    else:
        momentum_prev = momentum_val

    # Color LazyBear
    if momentum_val > 0:
        mom_color = "lime" if momentum_val > momentum_prev else "darkgreen"
    else:
        mom_color = "red" if momentum_val < momentum_prev else "maroon"

    # -- DIVERGENCIA PRECIO vs MOMENTUM --
    # Buscar en los ultimos 30 periodos
    div_lookback = min(30, len(close) - mom_length - 1)
    divergence = "NONE"
    div_score_adj = 0.0

    if div_lookback > 10:
        # Calcular momentum en ventana deslizante
        mom_series = []
        price_series = close.iloc[-(div_lookback + mom_length):].values
        delta_full = delta.iloc[-(div_lookback + mom_length):].dropna()

        for i in range(div_lookback):
            end_idx = len(delta_full) - div_lookback + i + 1
            start_idx = max(end_idx - mom_length, 0)
            if end_idx - start_idx < mom_length:
                mom_series.append(np.nan)
                continue
            chunk = delta_full.iloc[start_idx:end_idx]
            x = np.arange(len(chunk), dtype=float)
            y = chunk.values.astype(float)
            c = np.polyfit(x, y, 1)
            mom_series.append(float(c[0] * (len(x)-1) + c[1]))

        mom_arr = np.array(mom_series)
        price_arr = close.iloc[-div_lookback:].values.astype(float)

        # Buscar picos en precio y en momentum
        valid = ~np.isnan(mom_arr)
        if valid.sum() > 5:
            # Simplificado: comparar primera mitad vs segunda mitad
            half = len(price_arr) // 2
            price_first_max = float(np.max(price_arr[:half]))
            price_second_max = float(np.max(price_arr[half:]))
            mom_valid = mom_arr[valid]
            mom_first_half = mom_valid[:len(mom_valid)//2]
            mom_second_half = mom_valid[len(mom_valid)//2:]

            if len(mom_first_half) > 0 and len(mom_second_half) > 0:
                mom_first_max = float(np.max(mom_first_half))
                mom_second_max = float(np.max(mom_second_half))

                price_first_min = float(np.min(price_arr[:half]))
                price_second_min = float(np.min(price_arr[half:]))
                mom_first_min = float(np.min(mom_first_half))
                mom_second_min = float(np.min(mom_second_half))

                # Divergencia bajista: precio hace higher high, momentum lower high
                if (price_second_max > price_first_max * 1.005 and
                    mom_second_max < mom_first_max * 0.8):
                    divergence = "BEARISH_DIV"
                    div_score_adj = -0.4  # penalizar score hacia abajo

                # Divergencia alcista: precio hace lower low, momentum higher low
                elif (price_second_min < price_first_min * 0.995 and
                      mom_second_min > mom_first_min * 0.8):
                    divergence = "BULLISH_DIV"
                    div_score_adj = +0.4  # empujar score hacia arriba

    # -- Score final --
    mom_normalized = momentum_val / float(close.iloc[-1]) * 100

    if squeeze_on:
        score = float(np.clip(mom_normalized * 5, -0.7, 0.7))
        confidence = 0.5 + abs(score) * 0.3
    else:
        score = float(np.clip(mom_normalized * 8, -1.0, 1.0))
        if mom_color in ("lime", "red"):
            score *= 1.2
        confidence = min(abs(score), 1.0)

    # Aplicar divergencia
    score = float(np.clip(score + div_score_adj, -1, 1))
    if divergence != "NONE":
        confidence = min(confidence + 0.2, 1.0)

    return {
        "score": round(score, 3),
        "squeeze_on": squeeze_on,
        "momentum": round(momentum_val, 4),
        "mom_color": mom_color,
        "divergence": divergence,
        "confidence": round(confidence, 3),
    }


# ============================================================================
# INDICATOR 2: ADX
# ============================================================================

def compute_adx(high, low, close, period=14) -> dict:
    if len(close) < period * 3:
        return {"score": 0.0, "adx": 0.0, "plus_di": 0.0, "minus_di": 0.0,
                "trending": False, "confidence": 0.0}

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    cur_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
    cur_plus = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0
    cur_minus = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0

    trending = cur_adx > 25
    di_score = (cur_plus - cur_minus) / max(cur_plus + cur_minus, 1)

    if trending:
        score = di_score * min(cur_adx / 40, 1.0)
        adx_recent = adx.iloc[-5:].dropna()
        if len(adx_recent) >= 3 and float(adx_recent.iloc[-1]) > float(adx_recent.iloc[0]):
            score *= 1.15
        confidence = min(cur_adx / 50, 1.0)
    else:
        score = di_score * 0.3
        confidence = 0.2

    return {
        "score": round(float(np.clip(score, -1, 1)), 3),
        "adx": round(cur_adx, 1),
        "plus_di": round(cur_plus, 1),
        "minus_di": round(cur_minus, 1),
        "trending": trending,
        "confidence": round(confidence, 3),
    }


# ============================================================================
# INDICATOR 3: EMA 10/55
# ============================================================================

def compute_ema_signal(close, fast=10, slow=55) -> dict:
    if len(close) < slow + 10:
        return {"score": 0.0, "cross": "NEUTRAL", "distance_pct": 0.0, "confidence": 0.0}

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    cur_fast = float(ema_fast.iloc[-1])
    cur_slow = float(ema_slow.iloc[-1])
    cur_price = float(close.iloc[-1])

    distance_pct = (cur_fast - cur_slow) / cur_slow * 100

    # Cross en ultimas 5 barras
    recent_diff = (ema_fast - ema_slow).iloc[-5:]
    cross_type = "NONE"
    for i in range(1, len(recent_diff)):
        if recent_diff.iloc[i-1] < 0 and recent_diff.iloc[i] > 0:
            cross_type = "GOLDEN"
        elif recent_diff.iloc[i-1] > 0 and recent_diff.iloc[i] < 0:
            cross_type = "DEATH"

    fast_slope = float(ema_fast.iloc[-1] - ema_fast.iloc[-5]) / cur_price * 100
    price_vs_slow = (cur_price - cur_slow) / cur_slow * 100

    pos_score = float(np.clip(distance_pct / 3, -1, 1))
    slope_score = float(np.clip(fast_slope * 5, -1, 1))
    pvs_score = float(np.clip(price_vs_slow / 5, -1, 1))
    cross_score = 0.8 if cross_type == "GOLDEN" else (-0.8 if cross_type == "DEATH" else 0.0)

    score = 0.40 * pos_score + 0.30 * slope_score + 0.20 * pvs_score + 0.10 * cross_score
    score = float(np.clip(score, -1, 1))

    cross_label = f"BULL ({distance_pct:+.1f}%)" if cur_fast > cur_slow else f"BEAR ({distance_pct:+.1f}%)"

    return {
        "score": round(score, 3),
        "cross": cross_label,
        "distance_pct": round(distance_pct, 2),
        "cross_type": cross_type,
        "confidence": round(min(abs(score) * 1.3, 1.0), 3),
    }


# ============================================================================
# INDICATOR 4: Accumulation/Distribution
# ============================================================================

def compute_ad_line(high, low, close, volume, lookback=20) -> dict:
    if len(close) < lookback + 5:
        return {"score": 0.0, "divergence": "NONE", "confidence": 0.0}

    hl_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()

    def norm_slope(series, window=10):
        s = series.iloc[-window:]
        if len(s) < 3:
            return 0.0
        x = np.arange(len(s), dtype=float)
        y = (s - s.iloc[0]).values.astype(float)
        y_range = max(abs(y.max()), abs(y.min()), 1e-10)
        return float(np.polyfit(x, y / y_range, 1)[0])

    ad_slope = norm_slope(ad.iloc[-lookback:])
    price_slope = norm_slope(close.iloc[-lookback:])

    thr = 0.02
    if ad_slope > thr and price_slope < -thr:
        div, score = "BULL_DIV", 0.7 + min(abs(ad_slope - price_slope) * 3, 0.3)
    elif ad_slope < -thr and price_slope > thr:
        div, score = "BEAR_DIV", -(0.7 + min(abs(ad_slope - price_slope) * 3, 0.3))
    elif ad_slope > thr and price_slope > thr:
        div, score = "CONFIRM_UP", 0.4
    elif ad_slope < -thr and price_slope < -thr:
        div, score = "CONFIRM_DN", -0.4
    else:
        div, score = "NEUTRAL", float(np.clip(ad_slope * 5, -0.3, 0.3))

    return {
        "score": round(float(np.clip(score, -1, 1)), 3),
        "divergence": div,
        "confidence": round(min(abs(score), 1.0), 3),
    }


# ============================================================================
# INDICATOR 5: Volume Profile + POC
# ============================================================================

def compute_vpoc(close, volume, high, low, lookback=40, n_bins=20) -> dict:
    if len(close) < lookback:
        return {"score": 0.0, "poc": 0.0, "price_vs_poc": 0.0, "confidence": 0.0}

    c = close.iloc[-lookback:]
    v = volume.iloc[-lookback:]
    h = high.iloc[-lookback:]
    l = low.iloc[-lookback:]

    price_min, price_max = float(l.min()), float(h.max())
    if price_max == price_min:
        return {"score": 0.0, "poc": float(c.iloc[-1]), "price_vs_poc": 0.0, "confidence": 0.0}

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_vol = np.zeros(n_bins)

    for i in range(len(c)):
        bl, bh, bv = float(l.iloc[i]), float(h.iloc[i]), float(v.iloc[i])
        if bh == bl:
            idx = min(int((bl - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
            bin_vol[idx] += bv
        else:
            for j in range(n_bins):
                overlap = max(0, min(bh, bin_edges[j+1]) - max(bl, bin_edges[j]))
                if overlap > 0:
                    bin_vol[j] += bv * (overlap / (bh - bl))

    poc_bin = int(np.argmax(bin_vol))
    poc = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2)

    total = bin_vol.sum()
    if total == 0:
        return {"score": 0.0, "poc": poc, "price_vs_poc": 0.0, "confidence": 0.0}

    # Value Area 70%
    sorted_bins = np.argsort(bin_vol)[::-1]
    cumvol, va_bins = 0.0, set()
    for idx in sorted_bins:
        va_bins.add(idx)
        cumvol += bin_vol[idx]
        if cumvol >= 0.70 * total:
            break

    va_low = float(bin_edges[min(va_bins)])
    va_high = float(bin_edges[max(va_bins) + 1])
    cur_price = float(c.iloc[-1])
    price_vs_poc = (cur_price - poc) / poc * 100

    if cur_price > va_high:
        score = 0.3
    elif cur_price < va_low:
        score = -0.3
    elif cur_price > poc:
        score = 0.2 + 0.4 * ((cur_price - poc) / max(va_high - poc, 1e-10))
    else:
        score = -(0.2 + 0.4 * ((poc - cur_price) / max(poc - va_low, 1e-10)))

    return {
        "score": round(float(np.clip(score, -1, 1)), 3),
        "poc": round(poc, 2),
        "price_vs_poc": round(price_vs_poc, 2),
        "confidence": round(min(abs(score) * 1.5, 1.0), 3),
    }


# ============================================================================
# DIRECTIONAL FUSION
# ============================================================================

def compute_directional_score(close, high, low, volume, timeframe) -> dict:
    """Fusion de 5 indicadores en score direccional."""
    weights = WEIGHT_PROFILES.get(timeframe, WEIGHT_PROFILES["1D"])

    squeeze = compute_squeeze_momentum(close, high, low)
    adx = compute_adx(high, low, close)
    ema = compute_ema_signal(close)
    ad = compute_ad_line(high, low, close, volume)
    vpoc = compute_vpoc(close, volume, high, low)

    components = {
        "squeeze": (squeeze["score"], squeeze["confidence"], weights["squeeze"]),
        "adx": (adx["score"], adx["confidence"], weights["adx"]),
        "ema": (ema["score"], ema["confidence"], weights["ema"]),
        "ad": (ad["score"], ad["confidence"], weights["ad"]),
        "vpoc": (vpoc["score"], vpoc["confidence"], weights["vpoc"]),
    }

    w_sum, w_total = 0.0, 0.0
    for _, (sc, conf, w) in components.items():
        aw = w * (0.3 + 0.7 * conf)
        w_sum += aw * sc
        w_total += aw

    fused = float(np.clip(w_sum / max(w_total, 1e-10), -1, 1))

    scores = [squeeze["score"], adx["score"], ema["score"], ad["score"], vpoc["score"]]
    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)

    if pos >= 4:     agreement = "STRONG UP"
    elif neg >= 4:   agreement = "STRONG DOWN"
    elif pos >= 3:   agreement = "LEAN UP"
    elif neg >= 3:   agreement = "LEAN DOWN"
    else:            agreement = "MIXED"

    return {
        "score": round(fused, 3),
        "prob_up": round(float(np.clip(50 + fused * 50, 1, 99)), 1),
        "direction": "UP" if fused > 0 else "DOWN",
        "agreement": agreement,
        "sqz_s": squeeze["score"], "sqz_on": squeeze["squeeze_on"],
        "sqz_clr": squeeze["mom_color"], "sqz_div": squeeze["divergence"],
        "adx_s": adx["score"], "adx_v": adx["adx"], "adx_trend": adx["trending"],
        "ema_s": ema["score"], "ema_cross": ema["cross"],
        "ad_s": ad["score"], "ad_div": ad["divergence"],
        "vpoc_s": vpoc["score"], "vpoc_vs": vpoc["price_vs_poc"],
    }


# ============================================================================
# HARD MTF CASCADE
# ============================================================================

def apply_hard_cascade(lower_score: float, htf_result: dict) -> tuple:
    """
    Cascada dura: el TF superior impone techo/piso.

    1. Mezcla el score del LTF con el del HTF (peso 60/40 a favor del HTF)
    2. Aplica cap duro segun agreement del HTF

    Returns: (constrained_score, constrained_prob_up, cap_applied)
    """
    if htf_result is None:
        prob = 50 + lower_score * 50
        return lower_score, float(np.clip(prob, 1, 99)), "NONE"

    htf_score = htf_result["score"]
    htf_agree = htf_result["agreement"]

    # Paso 1: Blend con el HTF (el HTF pesa mas)
    # El TF inferior se mueve dentro de la ola del superior
    blended = 0.40 * lower_score + 0.60 * htf_score
    blended = float(np.clip(blended, -1, 1))

    prob_up = 50 + blended * 50
    prob_up = float(np.clip(prob_up, 1, 99))

    # Paso 2: Hard caps
    cap_applied = "NONE"
    caps = MTF_CAPS.get(htf_agree, {})

    if "max_prob_up" in caps and prob_up > caps["max_prob_up"]:
        prob_up = float(caps["max_prob_up"])
        blended = (prob_up - 50) / 50
        cap_applied = f"CAP_DN({caps['max_prob_up']})"

    if "min_prob_up" in caps and prob_up < caps["min_prob_up"]:
        prob_up = float(caps["min_prob_up"])
        blended = (prob_up - 50) / 50
        cap_applied = f"CAP_UP({caps['min_prob_up']})"

    return blended, prob_up, cap_applied


# ============================================================================
# RESAMPLE HELPERS (para cascada temporal)
# ============================================================================

RESAMPLE_MAP = {
    # Para cada TF, como resamplear al TF superior
    "4H": {"to_htf": "1D", "rule": "1D"},
    "1D": {"to_htf": "1W", "rule": "1W"},
}

def resample_to_htf(df_ohlcv: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resamplea OHLCV a un timeframe superior."""
    return df_ohlcv.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()


def compute_htf_signal_at_cutoff(df_train: pd.DataFrame, timeframe: str) -> dict:
    """
    Calcula la senal del TF SUPERIOR resampleando los datos del TF actual.
    Esto garantiza sincronizacion temporal perfecta con la cascada.

    Ej: si estamos en 1D, resamplea a 1W y computa indicadores semanales.
    """
    resample_info = RESAMPLE_MAP.get(timeframe)
    if resample_info is None:
        return None  # 1W no tiene TF superior

    htf_name = resample_info["to_htf"]
    rule = resample_info["rule"]

    df_htf = resample_to_htf(df_train, rule)

    if len(df_htf) < 60:  # minimo para calcular indicadores
        return None

    return compute_directional_score(
        df_htf["Close"].squeeze(), df_htf["High"].squeeze(),
        df_htf["Low"].squeeze(), df_htf["Volume"].squeeze(),
        htf_name,
    )


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_single_window(features_all, close_all, df_all, cutoff_idx,
                      horizon, n_regimes, z_score, config, timeframe):

    features_train = features_all.iloc[:cutoff_idx]
    close_train = close_all.iloc[:cutoff_idx]

    test_end = min(cutoff_idx + horizon, len(close_all))
    close_test = close_all.iloc[cutoff_idx:test_end]
    actual_horizon = len(close_test)
    if actual_horizon == 0:
        return None

    # HMM
    scaled_train, scaler = normalize_features(features_train)
    model, converged = train_hmm(scaled_train, n_regimes)
    regimes = decode_regimes(model, scaled_train)
    current_regime = regimes[-1]
    label_map, _, _ = label_regimes(model, n_regimes)
    last_close = float(close_train.iloc[-1])

    # Monte Carlo (rangos)
    range_data = compute_range_projection(
        model, current_regime, last_close,
        actual_horizon, z_score, scaler, N_SIMULATIONS
    )

    # OHLCV train
    train_end_loc = df_all.index.get_loc(close_train.index[-1])
    df_train = df_all.iloc[:train_end_loc + 1]

    # Direction V4b (indicadores del TF actual)
    dir_result = compute_directional_score(
        df_train["Close"].squeeze(), df_train["High"].squeeze(),
        df_train["Low"].squeeze(), df_train["Volume"].squeeze(),
        timeframe,
    )

    # CASCADA TEMPORAL: calcular la senal del TF superior
    # resampleando los datos del TF actual al momento del corte
    htf_result = compute_htf_signal_at_cutoff(df_train, timeframe)

    # Hard cascade
    v4b_score, v4b_prob_up, cap_applied = apply_hard_cascade(
        dir_result["score"], htf_result
    )
    v4b_dir = "UP" if v4b_score > 0 else "DOWN"

    # Eval
    real_prices = close_test.values
    real_final = float(real_prices[-1])
    actual_dir = "UP" if real_final > last_close else "DOWN"

    v1_dir = "UP" if range_data["prob_up"] > 50 else "DOWN"

    inside_iqr, inside_80, inside_full = 0, 0, 0
    for i in range(actual_horizon):
        p = float(real_prices[i])
        if range_data["tunnel_p25"][i] <= p <= range_data["tunnel_p75"][i]:
            inside_iqr += 1
        if range_data["tunnel_p10"][i] <= p <= range_data["tunnel_p90"][i]:
            inside_80 += 1
        if range_data["tunnel_lower"][i] <= p <= range_data["tunnel_upper"][i]:
            inside_full += 1

    htf_agree = htf_result["agreement"] if htf_result else "N/A"

    return {
        "cutoff_date": features_train.index[-1],
        "regime": label_map[current_regime],
        "real_pct": round((real_final - last_close) / last_close * 100, 2),
        "v1_dir": v1_dir, "v1_ok": v1_dir == actual_dir,
        "v4b_pup": round(v4b_prob_up, 1), "v4b_dir": v4b_dir,
        "v4b_ok": v4b_dir == actual_dir,
        "raw_s": round(dir_result["score"], 3),
        "final_s": round(v4b_score, 3),
        "cap": cap_applied,
        "agree": dir_result["agreement"],
        "htf_agree": htf_agree,
        "sqz_s": dir_result["sqz_s"], "sqz_div": dir_result["sqz_div"],
        "adx_s": dir_result["adx_s"], "ema_s": dir_result["ema_s"],
        "ad_s": dir_result["ad_s"], "ad_div": dir_result["ad_div"],
        "vpoc_s": dir_result["vpoc_s"],
        "iqr": round(inside_iqr / actual_horizon * 100, 1),
        "op80": round(inside_80 / actual_horizon * 100, 1),
        "full": round(inside_full / actual_horizon * 100, 1),
    }


def prepare_tf(ticker, tf):
    config = TIMEFRAME_CONFIG[tf]
    df = fetch_data(ticker, tf)
    if config["resample_rule"]:
        df = resample_ohlcv(df, config["resample_rule"])
    features = compute_features(df, config)
    close = df["Close"].squeeze().loc[features.index]
    return df, features, close, config


def run_cascade_backtest(ticker, n_regimes, n_windows, horizon, z_score):
    print(f"\n  Cargando datos...")
    tf_data = {}
    for tf in TIMEFRAMES_ORDER:
        df, feat, close, cfg = prepare_tf(ticker, tf)
        tf_data[tf] = {"df": df, "features": feat, "close": close, "config": cfg}
        print(f"  {tf}: {len(df)} barras | {len(feat)} features")

    for tf in TIMEFRAMES_ORDER:
        data = tf_data[tf]
        feat, close, df, cfg = data["features"], data["close"], data["df"], data["config"]

        actual_win = min(n_windows, (len(feat) - cfg["min_bars_required"]) // horizon)
        if actual_win < 1:
            print(f"\n  {tf}: SKIP (datos insuficientes)")
            continue

        htf_name = RESAMPLE_MAP.get(tf, {}).get("to_htf", "NONE")

        print(f"\n{'='*110}")
        if htf_name == "NONE":
            print(f"  {tf} | {actual_win} ventanas | TOP LEVEL (la ola)")
        else:
            print(f"  {tf} | {actual_win} ventanas | HTF cascade: "
                  f"data {tf} resampleada a {htf_name} para constraint")
        print(f"{'='*110}")

        print(f"  {'W':>2s} | {'Corte':>16s} | {'Reg':>8s} | {'Real':>7s} | "
              f"{'V1':>3s} | {'V4b':>3s} {'pUp':>5s} | {'Cap':>12s} | "
              f"{'HTF':>10s} | {'Agree':>10s} | "
              f"{'Sqz':>5s} {'Div':>10s} | {'ADX':>5s} | {'EMA':>5s} | "
              f"{'A/D':>5s} | {'VPOC':>5s} | {'IQR':>5s} | {'80%':>5s} | {'Full':>5s}")
        print(f"  {'-'*170}")

        results = []
        for w in range(actual_win):
            cutoff = len(feat) - (w + 1) * horizon
            if cutoff < cfg["min_bars_required"]:
                continue

            r = run_single_window(feat, close, df, cutoff, horizon,
                                   n_regimes, z_score, cfg, tf)
            if r is None:
                continue
            results.append(r)

            v1i = "OK" if r["v1_ok"] else "xx"
            v4i = "OK" if r["v4b_ok"] else "xx"

            print(f"  [{w+1:>2d}] | {str(r['cutoff_date'])[:16]:>16s} | "
                  f"{r['regime']:>8s} | {r['real_pct']:>+6.2f}% | "
                  f"{v1i:>3s} | {v4i:>3s} {r['v4b_pup']:>4.1f}% | "
                  f"{r['cap']:>12s} | "
                  f"{r['htf_agree']:>10s} | {r['agree']:>10s} | "
                  f"{r['sqz_s']:>+5.2f} {r['sqz_div']:>10s} | "
                  f"{r['adx_s']:>+5.2f} | {r['ema_s']:>+5.2f} | "
                  f"{r['ad_s']:>+5.2f} | {r['vpoc_s']:>+5.2f} | "
                  f"{r['iqr']:>4.1f}% | {r['op80']:>4.1f}% | {r['full']:>4.1f}%")

        tf_data[tf]["results"] = results

        if results:
            df_r = pd.DataFrame(results)
            n = len(df_r)
            v1a = df_r["v1_ok"].sum() / n * 100
            v4a = df_r["v4b_ok"].sum() / n * 100
            d = v4a - v1a
            ds = f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"

            print(f"\n  {'-'*60}")
            print(f"  RESUMEN {tf} ({n} ventanas)")
            print(f"  {'-'*60}")
            print(f"  V1 (MC only):     {v1a:>6.1f}%")
            print(f"  V4b (ind+cascade): {v4a:>6.1f}%  ({ds})")
            print(f"  IQR (P25-P75):    {df_r['iqr'].mean():>6.1f}%  (esperado ~50%)")
            print(f"  OP80 (P10-P90):   {df_r['op80'].mean():>6.1f}%  (esperado ~80%)")
            print(f"  Full (z=2):       {df_r['full'].mean():>6.1f}%  (esperado ~95%)")

    # Final table
    print(f"\n\n{'='*100}")
    print("  COMPARATIVA FINAL -- V1 vs V4b (Hard Cascade + Divergence)")
    print(f"{'='*100}")
    print(f"  {'TF':>4s} | {'Win':>3s} | {'V1':>6s} | {'V4b':>6s} | {'Delta':>6s} | {'IQR':>6s} | {'OP80':>6s} | {'Full':>6s}")
    print(f"  {'-'*60}")

    for tf in TIMEFRAMES_ORDER:
        res = tf_data[tf].get("results", [])
        if not res:
            continue
        df_r = pd.DataFrame(res)
        n = len(df_r)
        v1 = df_r["v1_ok"].sum() / n * 100
        v4 = df_r["v4b_ok"].sum() / n * 100
        d = v4 - v1
        ds = f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"
        print(f"  {tf:>4s} | {n:>3d} | {v1:>5.1f}% | {v4:>5.1f}% | {ds:>6s} | "
              f"{df_r['iqr'].mean():>5.1f}% | {df_r['op80'].mean():>5.1f}% | "
              f"{df_r['full'].mean():>5.1f}%")

    print(f"\n  Cascade temporal: datos del TF actual resampleados al TF superior")
    print(f"  Caps: STRONG_DOWN=max35% | LEAN_DOWN=max45% | LEAN_UP=min55% | STRONG_UP=min65%")
    print(f"\n  Backtest V4b finalizado.\n")


def main():
    print("\n" + "=" * 100)
    print("  WALK-FORWARD BACKTEST V4b -- Hard MTF Cascade + Momentum Divergence")
    print(f"  Par: {TICKER} | Regimenes: {N_REGIMES} | Horizonte: {HORIZON} | Ventanas: {N_WINDOWS}")
    print(f"  Cascade: 1W -> 1D -> 4H (HARD caps)")
    print(f"  New: Squeeze momentum divergence detection (bearish/bullish)")
    print("=" * 100)

    run_cascade_backtest(TICKER, N_REGIMES, N_WINDOWS, HORIZON, Z_SCORE)


if __name__ == "__main__":
    main()
