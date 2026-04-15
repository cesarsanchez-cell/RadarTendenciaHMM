"""
Walk-Forward Backtest — Monte Carlo Regime-Switching
====================================================
Evalua la precision de la proyeccion Monte Carlo del HMM retrocediendo
desde el presente en ventanas de 10 periodos.

Par: BTC/USDT | Temporalidades: 4H, 1D, 1W
"""

import warnings
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

# ── Importar funciones del app principal ──
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
HORIZON = 10          # periodos a proyectar en cada ventana
Z_SCORE = 2.0         # ~95% CI
N_SIMULATIONS = 10000


def run_single_window(features_all, close_all, cutoff_idx, horizon, n_regimes, z_score):
    """
    Entrena HMM hasta cutoff_idx, proyecta horizon periodos,
    compara con precios reales.
    """
    # ── Train split ──
    features_train = features_all.iloc[:cutoff_idx]
    close_train = close_all.iloc[:cutoff_idx]

    # ── Test split (los proximos horizon periodos) ──
    test_end = min(cutoff_idx + horizon, len(close_all))
    close_test = close_all.iloc[cutoff_idx:test_end]
    actual_horizon = len(close_test)

    if actual_horizon == 0:
        return None

    # ── Normalizar y entrenar ──
    scaled_train, scaler = normalize_features(features_train)
    model, converged = train_hmm(scaled_train, n_regimes)

    # ── Detectar regimen actual (ultimo del train) ──
    regimes = decode_regimes(model, scaled_train)
    current_regime = regimes[-1]

    # ── Label para mostrar ──
    label_map, color_map, sorted_order = label_regimes(model, n_regimes)
    current_label = label_map[current_regime]

    # ── Last close del train ──
    last_close = float(close_train.iloc[-1])

    # ── Monte Carlo projection ──
    range_data = compute_range_projection(
        model, current_regime, last_close,
        actual_horizon, z_score, scaler, N_SIMULATIONS
    )

    # ── Precios reales ──
    real_prices = close_test.values
    real_final = float(real_prices[-1])

    # ── Evaluar paso a paso ──
    inside_iqr = 0      # dentro de P25-P75
    inside_full = 0     # dentro del rango completo
    for i in range(actual_horizon):
        p = float(real_prices[i])
        p25 = range_data["tunnel_p25"][i]
        p75 = range_data["tunnel_p75"][i]
        low = range_data["tunnel_lower"][i]
        high = range_data["tunnel_upper"][i]

        if p25 <= p <= p75:
            inside_iqr += 1
        if low <= p <= high:
            inside_full += 1

    pct_inside_iqr = inside_iqr / actual_horizon * 100
    pct_inside_full = inside_full / actual_horizon * 100

    # ── Direccion ──
    predicted_direction = "UP" if range_data["prob_up"] > 50 else "DOWN"
    actual_direction = "UP" if real_final > last_close else "DOWN"
    direction_correct = predicted_direction == actual_direction

    # ── Error de mediana ──
    median_final = range_data["mid"]
    median_error_pct = (median_final - real_final) / last_close * 100

    # ── Retorno real ──
    real_return_pct = (real_final - last_close) / last_close * 100

    return {
        "cutoff_date": features_train.index[-1],
        "test_start": close_test.index[0],
        "test_end": close_test.index[-1],
        "regime": current_label,
        "converged": converged,
        "last_close": last_close,
        "real_final": real_final,
        "real_return_pct": round(real_return_pct, 2),
        "median_proj": round(median_final, 2),
        "median_error_pct": round(median_error_pct, 2),
        "prob_up": round(range_data["prob_up"], 1),
        "prob_down": round(range_data["prob_down"], 1),
        "predicted_dir": predicted_direction,
        "actual_dir": actual_direction,
        "dir_correct": direction_correct,
        "pct_inside_iqr": round(pct_inside_iqr, 1),
        "pct_inside_full": round(pct_inside_full, 1),
        "proj_upper": round(range_data["upper"], 2),
        "proj_lower": round(range_data["lower"], 2),
        "skew_ratio": round(range_data["skew_ratio"], 2),
    }


def run_backtest_for_timeframe(ticker, timeframe, n_regimes, n_windows, horizon, z_score):
    """Corre el walk-forward backtest completo para una temporalidad."""
    print(f"\n{'='*70}")
    print(f"  BACKTEST: {ticker} | {timeframe} | {n_regimes} regimenes | {horizon} periodos")
    print(f"{'='*70}")

    # ── Fetch data ──
    config = TIMEFRAME_CONFIG[timeframe]
    df = fetch_data(ticker, timeframe)

    if config["resample_rule"]:
        df = resample_ohlcv(df, config["resample_rule"])

    print(f"  Datos: {len(df)} barras | {df.index[0]} -> {df.index[-1]}")

    # ── Features sobre todo el dataset ──
    features_all = compute_features(df, config)
    close_all = df["Close"].squeeze().loc[features_all.index]

    print(f"  Features: {len(features_all)} filas (despues de dropna)")
    print(f"  Ventanas: {n_windows} x {horizon} periodos retrocediendo desde el final\n")

    results = []

    for w in range(n_windows):
        offset = (w + 1) * horizon
        cutoff_idx = len(features_all) - offset

        if cutoff_idx < config["min_bars_required"]:
            print(f"  [Window {w+1}] SKIP — insuficientes datos para entrenar ({cutoff_idx} < {config['min_bars_required']})")
            continue

        result = run_single_window(
            features_all, close_all, cutoff_idx,
            horizon, n_regimes, z_score
        )

        if result is None:
            print(f"  [Window {w+1}] SKIP — sin datos de test")
            continue

        results.append(result)

        dir_icon = "OK" if result["dir_correct"] else "MISS"
        print(f"  [Window {w+1}] Corte: {str(result['cutoff_date'])[:16]} | "
              f"Regimen: {result['regime']:>8s} | "
              f"Real: {result['real_return_pct']:>+6.2f}% | "
              f"Prob Up: {result['prob_up']:>5.1f}% | "
              f"Dir: {dir_icon:>4s} | "
              f"IQR: {result['pct_inside_iqr']:>5.1f}% | "
              f"Full: {result['pct_inside_full']:>5.1f}%")

    if not results:
        print("  No se pudieron ejecutar ventanas.")
        return None

    # ── Resumen ──
    df_results = pd.DataFrame(results)
    n = len(df_results)

    dir_accuracy = df_results["dir_correct"].sum() / n * 100
    avg_iqr = df_results["pct_inside_iqr"].mean()
    avg_full = df_results["pct_inside_full"].mean()
    avg_median_err = df_results["median_error_pct"].abs().mean()

    print(f"\n  {'-'*50}")
    print(f"  RESUMEN {timeframe} ({n} ventanas)")
    print(f"  {'-'*50}")
    print(f"  Precision direccional:      {dir_accuracy:>6.1f}%  (aciertos dir UP/DOWN)")
    print(f"  Contencion IQR (P25-P75):   {avg_iqr:>6.1f}%  (esperado ~50%)")
    print(f"  Contencion Full (z={z_score}):   {avg_full:>6.1f}%  (esperado ~95%)")
    print(f"  Error mediana (abs medio):  {avg_median_err:>6.2f}%")
    print(f"  {'-'*50}")

    return {
        "timeframe": timeframe,
        "n_windows": n,
        "dir_accuracy": round(dir_accuracy, 1),
        "avg_iqr_containment": round(avg_iqr, 1),
        "avg_full_containment": round(avg_full, 1),
        "avg_median_error": round(avg_median_err, 2),
        "details": df_results,
    }


def main():
    print("\n" + "=" * 70)
    print("  WALK-FORWARD BACKTEST -- Monte Carlo Regime-Switching HMM")
    print(f"  Par: {TICKER} | Regimenes: {N_REGIMES} | Horizonte: {HORIZON} periodos")
    print(f"  Ventanas: {N_WINDOWS} | Simulaciones: {N_SIMULATIONS} | z-score: {Z_SCORE}")
    print("=" * 70)

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

    # ── Tabla comparativa final ──
    if summaries:
        print(f"\n\n{'='*70}")
        print("  COMPARATIVA FINAL — Todas las temporalidades")
        print(f"{'='*70}")
        print(f"  {'TF':>4s} | {'Windows':>7s} | {'Dir Acc':>7s} | {'IQR Cont':>8s} | {'Full Cont':>9s} | {'Med Err':>7s}")
        print(f"  {'-'*4} | {'-'*7} | {'-'*7} | {'-'*8} | {'-'*9} | {'-'*7}")
        for s in summaries:
            print(f"  {s['timeframe']:>4s} | {s['n_windows']:>7d} | "
                  f"{s['dir_accuracy']:>6.1f}% | {s['avg_iqr_containment']:>7.1f}% | "
                  f"{s['avg_full_containment']:>8.1f}% | {s['avg_median_error']:>6.2f}%")

        print(f"\n  Interpretacion:")
        print(f"  - Dir Acc: % de veces que acerto la direccion (>50% = mejor que moneda)")
        print(f"  - IQR Cont: % del tiempo que el precio real estuvo en P25-P75 (ideal ~50%)")
        print(f"  - Full Cont: % del tiempo en rango completo z={Z_SCORE} (ideal ~95%)")
        print(f"  - Med Err: error absoluto medio de la mediana proyectada vs real")

    print("\n  Backtest finalizado.\n")


if __name__ == "__main__":
    main()
