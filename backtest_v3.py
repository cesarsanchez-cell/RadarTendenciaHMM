"""
Walk-Forward Backtest V3 — Volume + Price Structure Directional Engine
======================================================================
Mejora la prediccion direccional sin tocar los rangos del Monte Carlo.

Senales nuevas (todas derivadas de OHLCV existente):
  1. Volume-Price Divergence: detecta agotamiento de compradores/vendedores
  2. OBV Trend: On-Balance Volume como proxy de acumulacion/distribucion
  3. Price Structure: Higher-Highs/Lower-Lows como definicion pura de tendencia
  4. Volatility Squeeze: compresion de vol como anticipador de movimiento

Compara V1 (MC only) vs V3 (volume-enhanced) lado a lado.

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
TIMEFRAMES = ["4H", "1D", "1W"]
N_REGIMES = 3
N_WINDOWS = 8
HORIZON = 10
Z_SCORE = 2.0
N_SIMULATIONS = 10000

# Pesos de fusion por timeframe
# vol_price = Volume-Price Divergence + OBV
# structure = Price Structure (HH/HL/LH/LL)
# squeeze = Volatility Squeeze (timing)
# mc = Monte Carlo baseline
WEIGHT_PROFILES = {
    "4H":  {"mc": 0.15, "vol_price": 0.35, "structure": 0.35, "squeeze": 0.15},
    "1D":  {"mc": 0.15, "vol_price": 0.30, "structure": 0.35, "squeeze": 0.20},
    "1W":  {"mc": 0.10, "vol_price": 0.30, "structure": 0.35, "squeeze": 0.25},
}


# ============================================================================
# SIGNAL 1: Volume-Price Divergence
# ============================================================================

def compute_volume_price_divergence(close: pd.Series, volume: pd.Series,
                                     lookback: int = 10) -> dict:
    """
    Detecta divergencias entre precio y volumen.

    Divergencia bajista: precio sube pero volumen en las subidas decrece
      -> compradores agotandose -> reversal DOWN probable
    Divergencia alcista: precio baja pero volumen en las bajadas decrece
      -> vendedores agotandose -> reversal UP probable

    Tambien mide si el volumen confirma el movimiento (convergencia).

    Returns score [-1, 1]: positivo = volumen sugiere UP, negativo = DOWN
    """
    if len(close) < lookback + 5 or len(volume) < lookback + 5:
        return {"score": 0.0, "type": "NEUTRAL", "confidence": 0.0}

    recent_close = close.iloc[-lookback:]
    recent_vol = volume.iloc[-lookback:]

    # Separar barras alcistas y bajistas
    returns = recent_close.pct_change().dropna()
    recent_vol_aligned = recent_vol.iloc[-len(returns):]

    up_mask = returns > 0
    down_mask = returns < 0

    # Volumen promedio en barras alcistas vs bajistas
    vol_on_up = float(recent_vol_aligned[up_mask].mean()) if up_mask.any() else 0
    vol_on_down = float(recent_vol_aligned[down_mask].mean()) if down_mask.any() else 0

    # Evitar division por cero
    total_vol = vol_on_up + vol_on_down
    if total_vol == 0:
        return {"score": 0.0, "type": "NEUTRAL", "confidence": 0.0}

    # Ratio: >1 = mas volumen en subidas (bullish), <1 = mas en bajadas (bearish)
    vol_ratio = vol_on_up / max(vol_on_down, 1e-10)

    # Tendencia del precio en el lookback
    price_change = float(recent_close.iloc[-1] / recent_close.iloc[0] - 1)

    # Detectar divergencias
    # Precio sube + vol_ratio baja (< 0.8) = divergencia bajista
    # Precio baja + vol_ratio alto (> 1.2) = divergencia alcista
    divergence_type = "NEUTRAL"
    score = 0.0

    if price_change > 0.01:  # precio subio >1%
        if vol_ratio < 0.7:
            # Divergencia bajista fuerte: sube sin volumen
            score = -0.8
            divergence_type = "BEARISH_DIV"
        elif vol_ratio < 0.9:
            # Divergencia bajista leve
            score = -0.4
            divergence_type = "WEAK_BEAR_DIV"
        elif vol_ratio > 1.5:
            # Confirmacion alcista: sube con volumen
            score = 0.6
            divergence_type = "BULL_CONFIRM"
        else:
            score = 0.2  # sube, vol neutral -> leve bullish
            divergence_type = "NEUTRAL_UP"

    elif price_change < -0.01:  # precio bajo >1%
        if vol_ratio > 1.5:
            # Divergencia alcista fuerte: baja sin volumen bajista
            score = 0.8
            divergence_type = "BULLISH_DIV"
        elif vol_ratio > 1.1:
            # Divergencia alcista leve
            score = 0.4
            divergence_type = "WEAK_BULL_DIV"
        elif vol_ratio < 0.7:
            # Confirmacion bajista: baja con volumen
            score = -0.6
            divergence_type = "BEAR_CONFIRM"
        else:
            score = -0.2  # baja, vol neutral -> leve bearish
            divergence_type = "NEUTRAL_DOWN"

    else:  # precio flat
        # En rango, el volumen indica acumulacion o distribucion
        if vol_ratio > 1.3:
            score = 0.3
            divergence_type = "ACCUMULATION"
        elif vol_ratio < 0.7:
            score = -0.3
            divergence_type = "DISTRIBUTION"

    confidence = min(abs(score), 1.0)

    return {
        "score": round(float(np.clip(score, -1, 1)), 3),
        "type": divergence_type,
        "confidence": round(confidence, 3),
        "vol_ratio": round(vol_ratio, 2),
        "price_change_pct": round(price_change * 100, 2),
    }


# ============================================================================
# SIGNAL 2: OBV Trend
# ============================================================================

def compute_obv_trend(close: pd.Series, volume: pd.Series,
                      lookback: int = 20, slope_window: int = 10) -> dict:
    """
    On-Balance Volume: acumula volumen sumando en barras alcistas,
    restando en bajistas. La tendencia del OBV indica acumulacion/distribucion.

    Divergencia OBV-Price:
    - OBV sube + precio lateral/baja = acumulacion -> UP
    - OBV baja + precio lateral/sube = distribucion -> DOWN

    Returns score [-1, 1]
    """
    if len(close) < lookback + 5:
        return {"score": 0.0, "obv_slope": 0.0, "price_slope": 0.0, "confidence": 0.0}

    # Calcular OBV
    price_diff = close.diff()
    obv = pd.Series(0.0, index=close.index)
    obv.iloc[0] = 0
    for i in range(1, len(close)):
        if price_diff.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price_diff.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    # Tomar los ultimos lookback periodos
    obv_recent = obv.iloc[-lookback:]
    close_recent = close.iloc[-lookback:]

    # Normalizar ambos a [0, 1] para comparar slopes
    obv_norm = (obv_recent - obv_recent.min())
    obv_range = obv_recent.max() - obv_recent.min()
    if obv_range > 0:
        obv_norm = obv_norm / obv_range
    else:
        obv_norm = obv_norm * 0

    close_norm = (close_recent - close_recent.min())
    close_range = close_recent.max() - close_recent.min()
    if close_range > 0:
        close_norm = close_norm / close_range
    else:
        close_norm = close_norm * 0

    # Slope de los ultimos slope_window periodos (regresion lineal simple)
    sw = min(slope_window, len(obv_norm))
    x = np.arange(sw, dtype=float)

    obv_tail = obv_norm.iloc[-sw:].values.astype(float)
    close_tail = close_norm.iloc[-sw:].values.astype(float)

    # Pendiente via least squares
    if len(x) > 2 and np.std(x) > 0:
        obv_slope = float(np.polyfit(x, obv_tail, 1)[0])
        price_slope = float(np.polyfit(x, close_tail, 1)[0])
    else:
        obv_slope = 0.0
        price_slope = 0.0

    # Scoring basado en divergencia OBV vs precio
    # OBV sube + precio plano/baja = acumulacion (bullish)
    # OBV baja + precio plano/sube = distribucion (bearish)
    if obv_slope > 0.02 and price_slope < 0.01:
        # Acumulacion: OBV sube, precio no
        strength = min(obv_slope * 10, 1.0)
        score = 0.3 + 0.7 * strength
    elif obv_slope < -0.02 and price_slope > -0.01:
        # Distribucion: OBV baja, precio no
        strength = min(abs(obv_slope) * 10, 1.0)
        score = -(0.3 + 0.7 * strength)
    elif obv_slope > 0.02 and price_slope > 0.02:
        # Confirmacion alcista: ambos suben
        score = 0.4
    elif obv_slope < -0.02 and price_slope < -0.02:
        # Confirmacion bajista: ambos bajan
        score = -0.4
    else:
        score = 0.0

    score = float(np.clip(score, -1, 1))

    return {
        "score": round(score, 3),
        "obv_slope": round(obv_slope, 4),
        "price_slope": round(price_slope, 4),
        "confidence": round(abs(score), 3),
    }


# ============================================================================
# SIGNAL 3: Price Structure (Swing Analysis)
# ============================================================================

def compute_price_structure(high: pd.Series, low: pd.Series, close: pd.Series,
                             lookback: int = 20, swing_size: int = 3) -> dict:
    """
    Analiza la estructura de precio: Higher Highs / Higher Lows (uptrend)
    vs Lower Highs / Lower Lows (downtrend).

    Usa swing points (maximos/minimos locales) en vez de cada barra
    para filtrar ruido.

    Returns score [-1, 1]: +1 = uptrend claro, -1 = downtrend claro
    """
    if len(high) < lookback + swing_size * 2:
        return {"score": 0.0, "hh": 0, "hl": 0, "lh": 0, "ll": 0, "confidence": 0.0}

    h = high.iloc[-lookback:].values.astype(float)
    l = low.iloc[-lookback:].values.astype(float)

    # Encontrar swing highs y swing lows
    # Swing high: punto donde high[i] > high de las swing_size barras antes y despues
    swing_highs = []
    swing_lows = []

    for i in range(swing_size, len(h) - swing_size):
        # Swing high
        if h[i] == max(h[i-swing_size:i+swing_size+1]):
            swing_highs.append((i, h[i]))
        # Swing low
        if l[i] == min(l[i-swing_size:i+swing_size+1]):
            swing_lows.append((i, l[i]))

    # Contar patrones
    hh = 0  # higher highs
    lh = 0  # lower highs
    hl = 0  # higher lows
    ll = 0  # lower lows

    for j in range(1, len(swing_highs)):
        if swing_highs[j][1] > swing_highs[j-1][1]:
            hh += 1
        else:
            lh += 1

    for j in range(1, len(swing_lows)):
        if swing_lows[j][1] > swing_lows[j-1][1]:
            hl += 1
        else:
            ll += 1

    total_swings = hh + lh + hl + ll
    if total_swings == 0:
        return {"score": 0.0, "hh": 0, "hl": 0, "lh": 0, "ll": 0, "confidence": 0.0}

    # Score: (bullish patterns - bearish patterns) / total
    bullish = hh + hl
    bearish = lh + ll
    score = (bullish - bearish) / total_swings

    # Bonus: si los ultimos 2 swings son consistentes, reforzar
    recent_bias = 0
    if len(swing_highs) >= 2:
        if swing_highs[-1][1] > swing_highs[-2][1]:
            recent_bias += 0.15
        else:
            recent_bias -= 0.15
    if len(swing_lows) >= 2:
        if swing_lows[-1][1] > swing_lows[-2][1]:
            recent_bias += 0.15
        else:
            recent_bias -= 0.15

    score = float(np.clip(score + recent_bias, -1, 1))
    confidence = min(abs(score) * 1.2, 1.0)  # boost confidence un poco

    return {
        "score": round(score, 3),
        "hh": hh,
        "hl": hl,
        "lh": lh,
        "ll": ll,
        "n_swings": total_swings,
        "confidence": round(confidence, 3),
    }


# ============================================================================
# SIGNAL 4: Volatility Squeeze
# ============================================================================

def compute_volatility_squeeze(close: pd.Series, high: pd.Series, low: pd.Series,
                                atr_period: int = 14, lookback: int = 50) -> dict:
    """
    Detecta compresion de volatilidad (squeeze) que precede movimientos grandes.

    ATR Percentile: si el ATR actual esta en el percentil bajo de su historia
    reciente, estamos en squeeze.

    No predice direccion por si solo, pero amplifica la confianza de las
    otras senales: si hay squeeze + senal direccional, el movimiento
    sera mas fuerte.

    Returns:
    - squeeze_score [0, 1]: 1 = maxima compresion
    - expansion_bias [-1, 1]: si la vol esta expandiendo, en que direccion
    """
    if len(close) < lookback + atr_period:
        return {"squeeze_score": 0.0, "expansion_bias": 0.0, "atr_percentile": 50.0}

    # ATR (Average True Range)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs(),
    }).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    # ATR normalizado por precio (para comparabilidad)
    atr_pct = (atr / close * 100).dropna()

    if len(atr_pct) < lookback:
        return {"squeeze_score": 0.0, "expansion_bias": 0.0, "atr_percentile": 50.0}

    recent_atr = atr_pct.iloc[-lookback:]
    current_atr = float(atr_pct.iloc[-1])

    # Percentil del ATR actual vs historia reciente
    atr_percentile = float((recent_atr < current_atr).sum() / len(recent_atr) * 100)

    # Squeeze score: 1 cuando vol esta en minimos, 0 cuando esta en maximos
    squeeze_score = max(1 - atr_percentile / 50, 0)  # 0 percentile -> 1.0, 50+ -> 0

    # Expansion bias: si la vol esta expandiendo, medir en que direccion
    # (las ultimas 3-5 barras tuvieron subidas o bajadas)
    recent_returns = close.pct_change().iloc[-5:]
    if len(recent_returns.dropna()) >= 3:
        avg_return = float(recent_returns.mean())
        # Si vol expandiendo (atr_percentile > 60) + precio cayendo = bearish expansion
        # Si vol expandiendo + precio subiendo = bullish expansion
        if atr_percentile > 60:
            expansion_bias = float(np.clip(avg_return * 100, -1, 1))
        else:
            expansion_bias = 0.0
    else:
        expansion_bias = 0.0

    return {
        "squeeze_score": round(squeeze_score, 3),
        "expansion_bias": round(expansion_bias, 3),
        "atr_percentile": round(atr_percentile, 1),
    }


# ============================================================================
# DIRECTIONAL FUSION V3
# ============================================================================

def compute_v3_direction(mc_prob_up: float, vol_div: dict, obv: dict,
                          structure: dict, squeeze: dict,
                          timeframe: str = "1D") -> dict:
    """
    Fusiona las 4 senales nuevas + MC en una prediccion direccional.

    La volatility squeeze NO vota direccion, pero AMPLIFICA la confianza
    de las otras senales: si hay squeeze, el movimiento sera mas fuerte,
    entonces confiamos mas en la direccion que indican las otras senales.
    """
    weights = WEIGHT_PROFILES.get(timeframe, WEIGHT_PROFILES["1D"])

    # Scores individuales
    mc_score = (mc_prob_up - 50) / 50  # [0,100] -> [-1,1]

    # Volume-Price score combinado (divergencia + OBV)
    vol_score = 0.55 * vol_div["score"] + 0.45 * obv["score"]
    vol_score = float(np.clip(vol_score, -1, 1))

    # Price structure score
    struct_score = structure["score"]

    # Squeeze amplification factor
    # squeeze_score [0,1]: 0 = no squeeze, 1 = max squeeze
    # Cuando hay squeeze, amplificar las senales (el movimiento sera grande)
    squeeze_amplifier = 1.0 + squeeze["squeeze_score"] * 0.5  # 1.0 a 1.5x

    # Si vol esta expandiendo, el expansion_bias da direccion adicional
    squeeze_dir = squeeze["expansion_bias"]

    # Weighted base score (sin squeeze aun)
    w_mc = weights["mc"]
    w_vol = weights["vol_price"]
    w_struct = weights["structure"]
    w_squeeze = weights["squeeze"]

    # Ajustar pesos por confianza
    vol_conf = max(vol_div["confidence"], obv["confidence"])
    struct_conf = structure["confidence"]

    w_vol_adj = w_vol * (0.4 + 0.6 * vol_conf)
    w_struct_adj = w_struct * (0.4 + 0.6 * struct_conf)
    w_total = w_mc + w_vol_adj + w_struct_adj + w_squeeze

    base_score = (
        w_mc * mc_score +
        w_vol_adj * vol_score +
        w_struct_adj * struct_score +
        w_squeeze * squeeze_dir
    ) / w_total

    # Aplicar squeeze amplifier
    final_score = base_score * squeeze_amplifier
    final_score = float(np.clip(final_score, -1, 1))

    # Convertir a probabilidad
    enhanced_prob_up = 50 + final_score * 50
    enhanced_prob_up = float(np.clip(enhanced_prob_up, 1, 99))

    direction = "UP" if enhanced_prob_up > 50 else "DOWN"

    # Agreement check
    signs = [np.sign(mc_score), np.sign(vol_score), np.sign(struct_score)]
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    if pos == 3:
        agreement = "STRONG UP"
    elif neg == 3:
        agreement = "STRONG DOWN"
    elif pos >= 2:
        agreement = "LEAN UP"
    elif neg >= 2:
        agreement = "LEAN DOWN"
    else:
        agreement = "MIXED"

    return {
        "direction": direction,
        "enhanced_prob_up": round(enhanced_prob_up, 1),
        "final_score": round(final_score, 3),
        "mc_score": round(mc_score, 3),
        "vol_score": round(vol_score, 3),
        "struct_score": round(struct_score, 3),
        "squeeze_amp": round(squeeze_amplifier, 2),
        "squeeze_dir": round(squeeze_dir, 3),
        "agreement": agreement,
    }


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_single_window(features_all, close_all, df_all, cutoff_idx,
                      horizon, n_regimes, z_score, config, timeframe):
    """Entrena HMM, corre MC + V3 signals, compara con real."""

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
    label_map, color_map, sorted_order = label_regimes(model, n_regimes)
    current_label = label_map[current_regime]

    last_close = float(close_train.iloc[-1])

    # -- Monte Carlo (rangos, no se toca) --
    range_data = compute_range_projection(
        model, current_regime, last_close,
        actual_horizon, z_score, scaler, N_SIMULATIONS
    )

    # -- Obtener OHLCV del train para las senales --
    # df_all tiene index completo con OHLCV
    # Necesitamos alinear al index de close_train
    train_end_loc = df_all.index.get_loc(close_train.index[-1])
    df_train = df_all.iloc[:train_end_loc + 1]
    close_full = df_train["Close"].squeeze()
    high_full = df_train["High"].squeeze()
    low_full = df_train["Low"].squeeze()
    volume_full = df_train["Volume"].squeeze()

    # -- SIGNAL 1: Volume-Price Divergence --
    vol_div = compute_volume_price_divergence(close_full, volume_full)

    # -- SIGNAL 2: OBV Trend --
    obv = compute_obv_trend(close_full, volume_full)

    # -- SIGNAL 3: Price Structure --
    structure = compute_price_structure(high_full, low_full, close_full)

    # -- SIGNAL 4: Volatility Squeeze --
    squeeze = compute_volatility_squeeze(close_full, high_full, low_full)

    # -- FUSION V3 --
    v3 = compute_v3_direction(
        range_data["prob_up"], vol_div, obv, structure, squeeze, timeframe
    )

    # -- Precios reales --
    real_prices = close_test.values
    real_final = float(real_prices[-1])

    # -- Evaluar rangos --
    inside_iqr = 0
    inside_full = 0
    for i in range(actual_horizon):
        p = float(real_prices[i])
        if range_data["tunnel_p25"][i] <= p <= range_data["tunnel_p75"][i]:
            inside_iqr += 1
        if range_data["tunnel_lower"][i] <= p <= range_data["tunnel_upper"][i]:
            inside_full += 1

    pct_inside_iqr = inside_iqr / actual_horizon * 100
    pct_inside_full = inside_full / actual_horizon * 100

    # -- Evaluar direccion --
    actual_dir = "UP" if real_final > last_close else "DOWN"
    v1_dir = "UP" if range_data["prob_up"] > 50 else "DOWN"
    v1_correct = v1_dir == actual_dir
    v3_dir = v3["direction"]
    v3_correct = v3_dir == actual_dir

    real_return_pct = (real_final - last_close) / last_close * 100

    return {
        "cutoff_date": features_train.index[-1],
        "regime": current_label,
        "real_return_pct": round(real_return_pct, 2),
        # V1
        "v1_prob_up": round(range_data["prob_up"], 1),
        "v1_dir": v1_dir,
        "v1_correct": v1_correct,
        # V3
        "v3_prob_up": v3["enhanced_prob_up"],
        "v3_dir": v3_dir,
        "v3_correct": v3_correct,
        "v3_score": v3["final_score"],
        "agreement": v3["agreement"],
        # Components
        "mc_s": v3["mc_score"],
        "vol_s": v3["vol_score"],
        "struct_s": v3["struct_score"],
        "sqz_amp": v3["squeeze_amp"],
        "vol_type": vol_div["type"],
        "vol_ratio": vol_div.get("vol_ratio", 0),
        "atr_pctl": squeeze["atr_percentile"],
        "struct_swings": structure.get("n_swings", 0),
        # Ranges (unchanged)
        "pct_inside_iqr": round(pct_inside_iqr, 1),
        "pct_inside_full": round(pct_inside_full, 1),
    }


def run_backtest_for_timeframe(ticker, timeframe, n_regimes, n_windows, horizon, z_score):
    """Corre el walk-forward backtest V1 vs V3."""
    print(f"\n{'='*90}")
    print(f"  BACKTEST V3: {ticker} | {timeframe} | {n_regimes} reg | {horizon} periodos")
    weights = WEIGHT_PROFILES.get(timeframe, WEIGHT_PROFILES["1D"])
    print(f"  Pesos: MC={weights['mc']} | VolPrice={weights['vol_price']} | "
          f"Structure={weights['structure']} | Squeeze={weights['squeeze']}")
    print(f"{'='*90}")

    config = TIMEFRAME_CONFIG[timeframe]
    df = fetch_data(ticker, timeframe)

    if config["resample_rule"]:
        df = resample_ohlcv(df, config["resample_rule"])

    print(f"  Datos: {len(df)} barras | {df.index[0]} -> {df.index[-1]}")

    features_all = compute_features(df, config)
    close_all = df["Close"].squeeze().loc[features_all.index]

    print(f"  Features: {len(features_all)} filas")
    print(f"  Ventanas: {n_windows} x {horizon} periodos\n")

    print(f"  {'W':>2s} | {'Corte':>16s} | {'Reg':>8s} | {'Real':>7s} | "
          f"{'V1':>4s} | {'V3':>4s} {'pUp':>5s} | {'Agree':>10s} | "
          f"{'VolType':>14s} | {'VR':>4s} | {'Str':>5s} | {'ATR%':>4s} | "
          f"{'IQR':>5s} | {'Full':>5s}")
    print(f"  {'-'*120}")

    results = []

    for w in range(n_windows):
        offset = (w + 1) * horizon
        cutoff_idx = len(features_all) - offset

        if cutoff_idx < config["min_bars_required"]:
            print(f"  [{w+1:>2d}] SKIP - datos insuficientes ({cutoff_idx} < {config['min_bars_required']})")
            continue

        result = run_single_window(
            features_all, close_all, df, cutoff_idx,
            horizon, n_regimes, z_score, config, timeframe
        )

        if result is None:
            continue

        results.append(result)

        v1i = "OK" if result["v1_correct"] else "xx"
        v3i = "OK" if result["v3_correct"] else "xx"

        print(f"  [{w+1:>2d}] | {str(result['cutoff_date'])[:16]:>16s} | "
              f"{result['regime']:>8s} | {result['real_return_pct']:>+6.2f}% | "
              f"{v1i:>4s} | {v3i:>4s} {result['v3_prob_up']:>4.1f}% | "
              f"{result['agreement']:>10s} | "
              f"{result['vol_type']:>14s} | {result['vol_ratio']:>4.1f} | "
              f"{result['struct_s']:>+5.2f} | {result['atr_pctl']:>4.0f} | "
              f"{result['pct_inside_iqr']:>4.1f}% | {result['pct_inside_full']:>4.1f}%")

    if not results:
        print("  No se pudieron ejecutar ventanas.")
        return None

    df_r = pd.DataFrame(results)
    n = len(df_r)

    v1_acc = df_r["v1_correct"].sum() / n * 100
    v3_acc = df_r["v3_correct"].sum() / n * 100
    avg_iqr = df_r["pct_inside_iqr"].mean()
    avg_full = df_r["pct_inside_full"].mean()

    delta = v3_acc - v1_acc
    d_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"

    print(f"\n  {'-'*60}")
    print(f"  RESUMEN {timeframe} ({n} ventanas)")
    print(f"  {'-'*60}")
    print(f"  V1 Dir accuracy (MC only):      {v1_acc:>6.1f}%")
    print(f"  V3 Dir accuracy (vol+struct):   {v3_acc:>6.1f}%  ({d_str})")
    print(f"  Contencion IQR (P25-P75):       {avg_iqr:>6.1f}%  (esperado ~50%)")
    print(f"  Contencion Full (z={z_score}):       {avg_full:>6.1f}%  (esperado ~95%)")
    print(f"  {'-'*60}")

    print(f"\n  Senales promedio:")
    print(f"    MC score:         {df_r['mc_s'].mean():>+.3f}")
    print(f"    Volume score:     {df_r['vol_s'].mean():>+.3f}")
    print(f"    Structure score:  {df_r['struct_s'].mean():>+.3f}")
    print(f"    Squeeze amp avg:  {df_r['sqz_amp'].mean():>.2f}x")
    print(f"    ATR percentile:   {df_r['atr_pctl'].mean():>.1f}")

    return {
        "timeframe": timeframe,
        "n": n,
        "v1_acc": round(v1_acc, 1),
        "v3_acc": round(v3_acc, 1),
        "delta": round(delta, 1),
        "avg_iqr": round(avg_iqr, 1),
        "avg_full": round(avg_full, 1),
    }


def main():
    print("\n" + "=" * 90)
    print("  WALK-FORWARD BACKTEST V3 -- Volume + Price Structure Engine")
    print(f"  Par: {TICKER} | Regimenes: {N_REGIMES} | Horizonte: {HORIZON}")
    print(f"  Senales: Vol-Price Divergence | OBV Trend | Price Structure | Vol Squeeze")
    print("=" * 90)

    summaries = []

    for tf in TIMEFRAMES:
        try:
            summary = run_backtest_for_timeframe(
                TICKER, tf, N_REGIMES, N_WINDOWS, HORIZON, Z_SCORE
            )
            if summary:
                summaries.append(summary)
        except Exception as e:
            print(f"\n  ERROR en {tf}: {e}")
            import traceback
            traceback.print_exc()

    if summaries:
        print(f"\n\n{'='*90}")
        print("  COMPARATIVA FINAL -- V1 (MC only) vs V3 (Volume + Structure)")
        print(f"{'='*90}")
        print(f"  {'TF':>4s} | {'Win':>3s} | {'V1 Dir':>7s} | {'V3 Dir':>7s} | {'Delta':>6s} | {'IQR':>6s} | {'Full':>6s}")
        print(f"  {'-'*4} | {'-'*3} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*6}")
        for s in summaries:
            d = s["delta"]
            ds = f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"
            print(f"  {s['timeframe']:>4s} | {s['n']:>3d} | "
                  f"{s['v1_acc']:>6.1f}% | {s['v3_acc']:>6.1f}% | "
                  f"{ds:>6s} | {s['avg_iqr']:>5.1f}% | {s['avg_full']:>5.1f}%")

        print(f"\n  Interpretacion:")
        print(f"  - V1: Monte Carlo prob_up puro")
        print(f"  - V3: Fusion de Volume-Price Divergence + OBV + Price Structure + Squeeze")
        print(f"  - Delta > 0 = V3 mejor que V1")
        print(f"  - IQR/Full deben mantenerse (rangos intactos)")

    print("\n  Backtest V3 finalizado.\n")


if __name__ == "__main__":
    main()
