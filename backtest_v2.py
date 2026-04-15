"""
Walk-Forward Backtest V2 — Directional Enhancement
===================================================
Mejora la prediccion direccional del Monte Carlo HMM con:
  1. Momentum Overlay: RSI + EMA cross como voto direccional independiente
  2. Regime Transition Detection: detecta agotamiento del regimen actual
     via predict_proba() y sesga la direccion en consecuencia

Compara V1 (baseline) vs V2 (enhanced) lado a lado.

Par: BTC/USDT | Temporalidades: 4H, 1D, 1W
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

from app import (
    TIMEFRAME_CONFIG,
    fetch_data,
    resample_ohlcv,
    compute_features,
    normalize_features,
    train_hmm,
    decode_regimes,
    label_regimes,
    _extract_regime_params,
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

# Pesos del sistema de scoring direccional (por timeframe)
# En TFs cortos el momentum tiene mas peso; en TFs largos la transicion
WEIGHT_PROFILES = {
    "4H":  {"mc": 0.25, "mom": 0.45, "trans": 0.30},
    "1D":  {"mc": 0.30, "mom": 0.35, "trans": 0.35},
    "1W":  {"mc": 0.20, "mom": 0.35, "trans": 0.45},
}
DEFAULT_WEIGHTS = {"mc": 0.25, "mom": 0.40, "trans": 0.35}

# RSI contrarian threshold: por encima de este valor en TFs largos,
# RSI se interpreta como senal de agotamiento (contrarian)
RSI_CONTRARIAN_THRESHOLD = 65
RSI_CONTRARIAN_TIMEFRAMES = {"1D", "1W"}  # solo en TFs largos


# ============================================================================
# MOMENTUM OVERLAY
# ============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI clasico de Wilder."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema_cross_signal(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Senal de cruce EMA: positivo = tendencia alcista, negativo = bajista.
    Normalizado a [-1, 1] dividiendo por el ATR-like spread.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    diff = ema_fast - ema_slow
    # Normalizar por precio para comparabilidad cross-timeframe
    signal = diff / close
    return signal


def compute_momentum_score(close: pd.Series, lookback: int = 5,
                           contrarian_mode: bool = False) -> dict:
    """
    Combina RSI + EMA cross en un score direccional [-1, 1].
    -1 = fuerte senal bajista, +1 = fuerte senal alcista, 0 = neutral.

    contrarian_mode: en TFs largos, RSI extremo (>65 o <35) se interpreta
    como senal de agotamiento en vez de confirmacion. Esto detecta techos
    y pisos donde el momentum esta sobreextendido.

    Returns dict con score, componentes y confianza.
    """
    rsi = compute_rsi(close)
    ema_signal = compute_ema_cross_signal(close)

    if len(rsi.dropna()) < 2 or len(ema_signal.dropna()) < 2:
        return {"score": 0.0, "rsi": 50.0, "ema_signal": 0.0, "confidence": 0.0,
                "rsi_score": 0.0, "ema_score": 0.0, "rsi_trend": 0.0, "contrarian": False}

    current_rsi = float(rsi.iloc[-1])
    current_ema = float(ema_signal.iloc[-1])

    # RSI score base: mapear [0,100] a [-1,1] con zona muerta en [40,60]
    if current_rsi > 60:
        rsi_score_raw = min((current_rsi - 50) / 30, 1.0)
    elif current_rsi < 40:
        rsi_score_raw = max((current_rsi - 50) / 30, -1.0)
    else:
        rsi_score_raw = 0.0

    # Contrarian mode: RSI > threshold = sobrecompra = senal DOWN
    contrarian_active = False
    if contrarian_mode:
        if current_rsi > RSI_CONTRARIAN_THRESHOLD:
            # Sobrecompra -> senal bajista (contrarian)
            rsi_score = -min((current_rsi - RSI_CONTRARIAN_THRESHOLD) / 20, 1.0)
            contrarian_active = True
        elif current_rsi < (100 - RSI_CONTRARIAN_THRESHOLD):
            # Sobreventa -> senal alcista (contrarian)
            rsi_score = min(((100 - RSI_CONTRARIAN_THRESHOLD) - current_rsi) / 20, 1.0)
            contrarian_active = True
        else:
            rsi_score = rsi_score_raw * 0.5  # en zona neutral, reducir peso
    else:
        rsi_score = rsi_score_raw

    # EMA cross score
    ema_recent = ema_signal.iloc[-60:].dropna()
    if len(ema_recent) > 10:
        ema_std = float(ema_recent.std())
        if ema_std > 0:
            ema_score = float(np.clip(current_ema / (2 * ema_std), -1, 1))
        else:
            ema_score = 0.0
    else:
        ema_score = 0.0

    # Tendencia del RSI
    rsi_recent = rsi.iloc[-lookback:]
    if len(rsi_recent) >= lookback:
        rsi_slope = float(rsi_recent.iloc[-1] - rsi_recent.iloc[0]) / lookback
        rsi_trend = float(np.clip(rsi_slope / 5, -1, 1))
    else:
        rsi_trend = 0.0

    # En contrarian mode, la tendencia del RSI bajando desde sobrecompra
    # refuerza la senal contrarian
    if contrarian_mode and contrarian_active:
        # 50% RSI contrarian + 25% EMA + 25% RSI trend
        score = 0.50 * rsi_score + 0.25 * ema_score + 0.25 * rsi_trend
    else:
        # 40% RSI + 35% EMA + 25% RSI trend
        score = 0.40 * rsi_score + 0.35 * ema_score + 0.25 * rsi_trend

    score = float(np.clip(score, -1, 1))
    confidence = abs(score)

    return {
        "score": score,
        "rsi": round(current_rsi, 1),
        "rsi_score": round(rsi_score, 3),
        "ema_score": round(ema_score, 3),
        "rsi_trend": round(rsi_trend, 3),
        "confidence": round(confidence, 3),
        "contrarian": contrarian_active,
    }


# ============================================================================
# REGIME TRANSITION DETECTION
# ============================================================================

def compute_transition_signal(model: GaussianHMM, features_scaled: np.ndarray,
                               current_regime: int, label_map: dict,
                               sorted_order: list, lookback: int = 10) -> dict:
    """
    Analiza la evolucion de predict_proba() en los ultimos `lookback` periodos
    para detectar si el regimen actual se esta agotando.

    Si P(regimen_actual) baja y P(regimen_opuesto) sube -> senal de cambio.

    Returns dict con:
    - regime_strength: [0,1] que tan fuerte es el regimen actual
    - transition_score: [-1,1] hacia donde esta transicionando
      (-1 = hacia bear, +1 = hacia bull)
    - confidence: [0,1]
    """
    proba = model.predict_proba(features_scaled)

    if len(proba) < lookback:
        lookback = len(proba)

    recent_proba = proba[-lookback:]

    # sorted_order[0] = bear, sorted_order[-1] = bull
    bear_idx = sorted_order[0]
    bull_idx = sorted_order[-1]

    # Probabilidad actual del regimen actual
    current_p = float(recent_proba[-1, current_regime])

    # Tendencia de P(current_regime) en los ultimos periodos
    p_current_series = recent_proba[:, current_regime]
    if len(p_current_series) >= 3:
        p_trend = float(p_current_series[-1] - p_current_series[0]) / lookback
    else:
        p_trend = 0.0

    # Tendencia de P(bull) - P(bear) -> indica hacia donde migra
    bull_trend = recent_proba[:, bull_idx]
    bear_trend = recent_proba[:, bear_idx]
    directional_now = float(bull_trend[-1] - bear_trend[-1])
    directional_start = float(bull_trend[0] - bear_trend[0])
    directional_change = directional_now - directional_start

    # Regime strength: P(current) actual, penalizada si viene bajando
    regime_strength = float(np.clip(current_p + p_trend * 5, 0, 1))

    # Transition score: hacia donde se esta moviendo la probabilidad
    # Positivo = hacia bull, negativo = hacia bear
    transition_score = float(np.clip(directional_change * 3, -1, 1))

    # Si el regimen actual se esta debilitando, la senal es mas fuerte
    weakening = p_trend < -0.02  # P(current) cayo >2% por periodo
    confidence = min(abs(transition_score) * (1.5 if weakening else 1.0), 1.0)

    return {
        "regime_strength": round(regime_strength, 3),
        "transition_score": round(transition_score, 3),
        "confidence": round(confidence, 3),
        "p_current": round(current_p, 3),
        "p_current_trend": round(p_trend, 4),
        "directional_now": round(directional_now, 3),
        "directional_change": round(directional_change, 3),
        "weakening": weakening,
    }


# ============================================================================
# DIRECTIONAL FUSION
# ============================================================================

def compute_enhanced_direction(mc_prob_up: float, momentum: dict,
                                transition: dict,
                                timeframe: str = "1D") -> dict:
    """
    Fusiona las 3 senales en una prediccion direccional mejorada:
    1. Monte Carlo prob_up (baseline)
    2. Momentum overlay score
    3. Regime transition detection

    Pesos adaptativos por timeframe y ajustados por confianza de cada senal.
    """
    # Pesos base por timeframe
    weights = WEIGHT_PROFILES.get(timeframe, DEFAULT_WEIGHTS)
    W_MC = weights["mc"]
    W_MOM = weights["mom"]
    W_TRANS = weights["trans"]

    # 1. Monte Carlo: convertir prob_up [0,100] a score [-1,1]
    mc_score = (mc_prob_up - 50) / 50

    # 2. Momentum: ya esta en [-1,1]
    mom_score = momentum["score"]

    # 3. Transition: ya esta en [-1,1]
    trans_score = transition["transition_score"]

    # Confidence-weighted: ajustar pesos por confianza de cada senal
    mom_conf = momentum["confidence"]
    trans_conf = transition["confidence"]

    w_mc = W_MC
    w_mom = W_MOM * (0.3 + 0.7 * mom_conf)
    w_trans = W_TRANS * (0.3 + 0.7 * trans_conf)
    w_total = w_mc + w_mom + w_trans

    weighted_score = (w_mc * mc_score + w_mom * mom_score + w_trans * trans_score) / w_total
    weighted_score = float(np.clip(weighted_score, -1, 1))

    # Convertir score a prob_up ajustado [0, 100]
    enhanced_prob_up = 50 + weighted_score * 50
    enhanced_prob_up = float(np.clip(enhanced_prob_up, 1, 99))

    # Direccion y confianza final
    direction = "UP" if enhanced_prob_up > 50 else "DOWN"
    final_confidence = abs(weighted_score)

    return {
        "direction": direction,
        "enhanced_prob_up": round(enhanced_prob_up, 1),
        "enhanced_prob_down": round(100 - enhanced_prob_up, 1),
        "weighted_score": round(weighted_score, 3),
        "final_confidence": round(final_confidence, 3),
        "mc_score": round(mc_score, 3),
        "mom_score": round(mom_score, 3),
        "trans_score": round(trans_score, 3),
        "agreement": _check_agreement(mc_score, mom_score, trans_score),
    }


def _check_agreement(mc, mom, trans) -> str:
    """Verifica si las 3 senales estan de acuerdo."""
    signs = [np.sign(mc), np.sign(mom), np.sign(trans)]
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    if pos == 3:
        return "STRONG UP"
    elif neg == 3:
        return "STRONG DOWN"
    elif pos >= 2:
        return "LEAN UP"
    elif neg >= 2:
        return "LEAN DOWN"
    return "MIXED"


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_single_window(features_all, close_all, df_all, cutoff_idx,
                      horizon, n_regimes, z_score, config, timeframe):
    """
    Entrena HMM hasta cutoff_idx, corre Monte Carlo + enhancements,
    compara con precios reales.
    """
    # -- Train / Test split --
    features_train = features_all.iloc[:cutoff_idx]
    close_train = close_all.iloc[:cutoff_idx]
    # close completo hasta cutoff para calcular indicadores
    close_full_train = df_all["Close"].squeeze().iloc[:close_all.index.get_loc(close_train.index[-1]) + 1]

    test_end = min(cutoff_idx + horizon, len(close_all))
    close_test = close_all.iloc[cutoff_idx:test_end]
    actual_horizon = len(close_test)

    if actual_horizon == 0:
        return None

    # -- Normalizar y entrenar --
    scaled_train, scaler = normalize_features(features_train)
    model, converged = train_hmm(scaled_train, n_regimes)

    # -- Detectar regimen actual --
    regimes = decode_regimes(model, scaled_train)
    current_regime = regimes[-1]
    label_map, color_map, sorted_order = label_regimes(model, n_regimes)
    current_label = label_map[current_regime]

    last_close = float(close_train.iloc[-1])

    # -- Monte Carlo projection (no se toca, genera rangos) --
    range_data = compute_range_projection(
        model, current_regime, last_close,
        actual_horizon, z_score, scaler, N_SIMULATIONS
    )

    # -- ENHANCEMENT 1: Momentum Overlay --
    use_contrarian = timeframe in RSI_CONTRARIAN_TIMEFRAMES
    momentum = compute_momentum_score(close_full_train, contrarian_mode=use_contrarian)

    # -- ENHANCEMENT 2: Regime Transition Detection --
    transition = compute_transition_signal(
        model, scaled_train, current_regime, label_map, sorted_order
    )

    # -- FUSION: Enhanced direction --
    enhanced = compute_enhanced_direction(
        range_data["prob_up"], momentum, transition, timeframe=timeframe
    )

    # -- Precios reales --
    real_prices = close_test.values
    real_final = float(real_prices[-1])

    # -- Evaluar rangos (identico a V1, no se toca) --
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

    # -- Evaluar direccion V1 (baseline) --
    v1_dir = "UP" if range_data["prob_up"] > 50 else "DOWN"
    actual_dir = "UP" if real_final > last_close else "DOWN"
    v1_correct = v1_dir == actual_dir

    # -- Evaluar direccion V2 (enhanced) --
    v2_dir = enhanced["direction"]
    v2_correct = v2_dir == actual_dir

    real_return_pct = (real_final - last_close) / last_close * 100

    return {
        "cutoff_date": features_train.index[-1],
        "test_start": close_test.index[0],
        "test_end": close_test.index[-1],
        "regime": current_label,
        "last_close": last_close,
        "real_final": real_final,
        "real_return_pct": round(real_return_pct, 2),
        # V1 baseline
        "v1_prob_up": round(range_data["prob_up"], 1),
        "v1_dir": v1_dir,
        "v1_correct": v1_correct,
        # V2 enhanced
        "v2_prob_up": enhanced["enhanced_prob_up"],
        "v2_dir": v2_dir,
        "v2_correct": v2_correct,
        "v2_score": enhanced["weighted_score"],
        "agreement": enhanced["agreement"],
        # Componentes
        "mc_score": enhanced["mc_score"],
        "mom_score": enhanced["mom_score"],
        "mom_rsi": momentum["rsi"],
        "trans_score": enhanced["trans_score"],
        "regime_strength": transition["regime_strength"],
        "weakening": transition["weakening"],
        "contrarian": momentum.get("contrarian", False),
        # Rangos (sin cambios)
        "pct_inside_iqr": round(pct_inside_iqr, 1),
        "pct_inside_full": round(pct_inside_full, 1),
    }


def run_backtest_for_timeframe(ticker, timeframe, n_regimes, n_windows, horizon, z_score):
    """Corre el walk-forward backtest V1 vs V2 para una temporalidad."""
    print(f"\n{'='*80}")
    print(f"  BACKTEST V2: {ticker} | {timeframe} | {n_regimes} regimenes | {horizon} periodos")
    print(f"{'='*80}")

    config = TIMEFRAME_CONFIG[timeframe]
    df = fetch_data(ticker, timeframe)

    if config["resample_rule"]:
        df = resample_ohlcv(df, config["resample_rule"])

    print(f"  Datos: {len(df)} barras | {df.index[0]} -> {df.index[-1]}")

    features_all = compute_features(df, config)
    close_all = df["Close"].squeeze().loc[features_all.index]

    print(f"  Features: {len(features_all)} filas")
    print(f"  Ventanas: {n_windows} x {horizon} periodos\n")

    header = (
        f"  {'Win':>3s} | {'Corte':>16s} | {'Regimen':>8s} | "
        f"{'Real':>7s} | {'V1':>4s} {'pUp':>5s} | {'V2':>4s} {'pUp':>5s} | "
        f"{'Agree':>10s} | {'RSI':>5s} | {'IQR':>5s} | {'Full':>5s}"
    )
    print(header)
    print(f"  {'-'*len(header)}")

    results = []

    for w in range(n_windows):
        offset = (w + 1) * horizon
        cutoff_idx = len(features_all) - offset

        if cutoff_idx < config["min_bars_required"]:
            print(f"  [{w+1:>2d}] SKIP - insuficientes datos ({cutoff_idx} < {config['min_bars_required']})")
            continue

        result = run_single_window(
            features_all, close_all, df, cutoff_idx,
            horizon, n_regimes, z_score, config, timeframe
        )

        if result is None:
            continue

        results.append(result)

        v1_icon = "OK" if result["v1_correct"] else "xx"
        v2_icon = "OK" if result["v2_correct"] else "xx"

        print(f"  [{w+1:>2d}] | {str(result['cutoff_date'])[:16]:>16s} | "
              f"{result['regime']:>8s} | "
              f"{result['real_return_pct']:>+6.2f}% | "
              f"{v1_icon:>4s} {result['v1_prob_up']:>4.1f}% | "
              f"{v2_icon:>4s} {result['v2_prob_up']:>4.1f}% | "
              f"{result['agreement']:>10s} | "
              f"{result['mom_rsi']:>5.1f} | "
              f"{result['pct_inside_iqr']:>4.1f}% | "
              f"{result['pct_inside_full']:>4.1f}%")

    if not results:
        print("  No se pudieron ejecutar ventanas.")
        return None

    df_r = pd.DataFrame(results)
    n = len(df_r)

    v1_acc = df_r["v1_correct"].sum() / n * 100
    v2_acc = df_r["v2_correct"].sum() / n * 100
    avg_iqr = df_r["pct_inside_iqr"].mean()
    avg_full = df_r["pct_inside_full"].mean()

    delta = v2_acc - v1_acc
    delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"

    print(f"\n  {'-'*60}")
    print(f"  RESUMEN {timeframe} ({n} ventanas)")
    print(f"  {'-'*60}")
    print(f"  V1 Dir accuracy (MC only):      {v1_acc:>6.1f}%")
    print(f"  V2 Dir accuracy (enhanced):     {v2_acc:>6.1f}%  ({delta_str})")
    print(f"  Contencion IQR (P25-P75):       {avg_iqr:>6.1f}%  (esperado ~50%)")
    print(f"  Contencion Full (z={z_score}):       {avg_full:>6.1f}%  (esperado ~95%)")
    print(f"  {'-'*60}")

    # Detalle de senales
    print(f"\n  Detalle de senales promedio:")
    print(f"    MC score avg:       {df_r['mc_score'].mean():>+.3f}")
    print(f"    Momentum score avg: {df_r['mom_score'].mean():>+.3f}")
    print(f"    Transition avg:     {df_r['trans_score'].mean():>+.3f}")
    print(f"    RSI avg:            {df_r['mom_rsi'].mean():>5.1f}")
    print(f"    Regime weakening:   {df_r['weakening'].sum()}/{n} ventanas")
    if 'contrarian' in df_r.columns:
        print(f"    RSI contrarian:     {df_r['contrarian'].sum()}/{n} ventanas")

    return {
        "timeframe": timeframe,
        "n_windows": n,
        "v1_accuracy": round(v1_acc, 1),
        "v2_accuracy": round(v2_acc, 1),
        "delta": round(delta, 1),
        "avg_iqr": round(avg_iqr, 1),
        "avg_full": round(avg_full, 1),
    }


def main():
    print("\n" + "=" * 80)
    print("  WALK-FORWARD BACKTEST V2 -- Enhanced Directional Prediction")
    print(f"  Par: {TICKER} | Regimenes: {N_REGIMES} | Horizonte: {HORIZON}")
    print(f"  Pesos por TF: {WEIGHT_PROFILES}")
    print("=" * 80)

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
        print(f"\n\n{'='*80}")
        print("  COMPARATIVA FINAL -- V1 (baseline) vs V2 (enhanced)")
        print(f"{'='*80}")
        print(f"  {'TF':>4s} | {'Win':>3s} | {'V1 Dir':>7s} | {'V2 Dir':>7s} | {'Delta':>6s} | {'IQR':>6s} | {'Full':>6s}")
        print(f"  {'-'*4} | {'-'*3} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*6}")
        for s in summaries:
            d = s["delta"]
            d_str = f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"
            print(f"  {s['timeframe']:>4s} | {s['n_windows']:>3d} | "
                  f"{s['v1_accuracy']:>6.1f}% | {s['v2_accuracy']:>6.1f}% | "
                  f"{d_str:>6s} | {s['avg_iqr']:>5.1f}% | {s['avg_full']:>5.1f}%")

        print(f"\n  Interpretacion:")
        print(f"  - V1 Dir: prediccion solo con Monte Carlo prob_up")
        print(f"  - V2 Dir: fusion MC + Momentum + Regime Transition")
        print(f"  - Delta: mejora de V2 sobre V1 (positivo = V2 mejor)")
        print(f"  - IQR/Full: NO deben cambiar (los rangos no se tocan)")

    print("\n  Backtest V2 finalizado.\n")


if __name__ == "__main__":
    main()
