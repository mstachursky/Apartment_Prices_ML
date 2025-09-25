import argparse
from pathlib import Path
import joblib
import sys
import datetime
import gc
import pandas as pd
import numpy as np
import json

# lokalne moduły
from src.utils import setup_dirs
from src.data_loader import load_and_dedup
from src.preprocessing import (
    basic_cleaning, remove_percentile_outliers_per_city,
    prepare_datasets
)
from src.modeling import (
    train_random_forest, train_xgb_with_loo, evaluate_model_simple
)
from src.evaluate import (
    analyze_errors_and_save, perform_eda_and_save, shap_analysis_and_save
)


def main(args):
    """
    Główna funkcja pipeline'u modelowania cen mieszkań.
    Wykonuje następujące kroki (z komunikatami logującymi postęp):
      1. Tworzy katalogi projektu.
      2. Wczytuje i łączy dane z plików CSV.
      3. Czyści dane i usuwa skrajne wartości.
      4. Przeprowadza EDA (opcjonalnie).
      5. Przygotowuje dane dla RandomForest i XGBoost.
      6. Dzieli dane na zbiory treningowe, walidacyjne i testowe.
      7. Trenuje i ewaluuje model RandomForest, zapisuje artefakty i raporty.
      8. Trenuje i ewaluuje model XGBoost z kodowaniem LOO, zapisuje artefakty i raporty.
      9. Przeprowadza analizę SHAP (opcjonalnie).
      10. Czyści pamięć i kończy pipeline.

    Parametry:
        args: Argumenty z parsera argparse.

    Zwraca:
        int: Kod zakończenia (0).
    """
    print("==> Pipeline start")
    print(f"Config: data_dir={args.data_dir}, full={args.full}, n_trials={args.n_trials}, save_plots={args.save_plots}")

    # 1) Tworzy katalogi pomocnicze
    print("-> Creating project directories (models, reports, logs)...")
    setup_dirs()
    print("   Directories ensured.")

    # 2) Wczytaj dane (obsługa braku plików)
    data_dir = Path(args.data_dir)
    print(f"-> Loading CSV files from: {data_dir}")
    try:
        df = load_and_dedup(data_dir, pattern='apartments_pl_*.csv')  # Wczytuje i łączy pliki CSV
        print(f"   Loaded data shape: {df.shape}")
    except FileNotFoundError as e:
        print("ERROR: No CSV files found:", e)
        return 2
    except Exception as e:
        print("ERROR: Unexpected error while loading data:", e)
        return 3

    print("Rows:", len(df), "Unique ids:", df['id'].nunique() if 'id' in df.columns else 'N/A')

    # 3) Czyszczenie danych: usuwanie skrajnych wartości i podstawowe czyszczenie
    print("-> Removing percentile outliers per city (price) ...")
    try:
        df = remove_percentile_outliers_per_city(df, target_col='price', lower_p=0.01, upper_p=0.95)
        print(f"   After outlier removal: {df.shape}")
    except Exception as e:
        print("ERROR during outlier removal:", e)
        return 4

    print("-> Running basic cleaning ...")
    try:
        df = basic_cleaning(df)
        print(f"   After basic cleaning: {df.shape}")
    except Exception as e:
        print("ERROR during basic cleaning:", e)
        return 5

    # 4) Eksploracyjna analiza danych (EDA) i zapis raportów, jeśli wybrano tryb pełny
    if args.full and args.save_plots:
        print("-> Performing EDA and saving plots/reports (full run)...")
        try:
            perform_eda_and_save(df, save_prefix='eda_full')
            print("   EDA saved.")
        except Exception as e:
            print("Warning: EDA failed:", e)

    # 5) Przygotowanie danych dla RandomForest i XGBoost
    print("-> Preparing datasets for RF and XGB ...")
    try:
        df_rf, df_xgb = prepare_datasets(df)
        print(f"   df_rf shape: {df_rf.shape}, df_xgb shape: {df_xgb.shape}")
    except Exception as e:
        print("ERROR preparing datasets:", e)
        return 6

    # 6) Podział danych dla RandomForest na train/val/test
    from sklearn.model_selection import train_test_split
    X = df_rf.drop(columns=['price'])
    y = df_rf['price']
    print("-> Splitting RF datasets (train/val/test)...")
    X_train_rf, X_temp_rf, y_train_rf, y_temp_rf = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val_rf, X_test_rf, y_val_rf, y_test_rf = train_test_split(X_temp_rf, y_temp_rf, test_size=0.5, random_state=42)
    print("   RF shapes:", X_train_rf.shape, X_val_rf.shape, X_test_rf.shape)

    # 7) Trening szybkiego modelu RandomForest na zbiorze treningowym
    print("-> Training quick RandomForest (n_estimators=100)...")
    try:
        rf_model = train_random_forest(X_train_rf, y_train_rf, n_estimators=100)
        print("   RF quick model trained.")
    except Exception as e:
        print("ERROR training RF quick:", e)
        return 7

    print("-> Evaluating RF quick model on test set...")
    try:
        y_pred_rf, rf_metrics = evaluate_model_simple(rf_model, X_test_rf, y_test_rf, scaler=None)
        print("   RF quick metrics:", rf_metrics)
    except Exception as e:
        print("ERROR evaluating RF quick model:", e)
        return 8

    # 8) Zapis artefaktów modelu RF
    print("-> Saving RF quick artifact to models/rf_quick.pkl ...")
    try:
        rf_art = {
            'model': rf_model,
            'feature_names': X_train_rf.columns.tolist(),
            'scaler': None,
            'metadata': {
                'trained_at': datetime.datetime.now().isoformat(),
                'metrics_train_only': rf_metrics
            }
        }
        joblib.dump(rf_art, 'models/rf_quick.pkl')
        print("   Saved RF quick artifact.")
    except Exception as e:
        print("Warning: Could not save RF quick artifact:", e)

    # Analiza błędów RF i zapis wykresów
    print("-> Analyzing RF errors and saving reports...")
    try:
        analyze_errors_and_save(y_test_rf, y_pred_rf, prefix='reports/rf_error')
        print("   RF error reports saved.")
    except Exception as e:
        print("Warning: RF error analysis failed:", e)

    # Pełny trening RF na train+val, zapis finalnego artefaktu i raportu (jeśli wybrano tryb pełny)
    if args.full:
        print("-> Full run: retraining RandomForest on train+val for final artifact...")
        try:
            X_combined_rf = pd.concat([X_train_rf, X_val_rf], ignore_index=True)
            y_combined_rf = pd.concat([y_train_rf, y_val_rf], ignore_index=True)

            rf_final = train_random_forest(X_combined_rf, y_combined_rf, n_estimators=200)
            print("   RF final model trained.")
            y_pred_rf_final, rf_final_metrics = evaluate_model_simple(rf_final, X_test_rf, y_test_rf, scaler=None)
            print("   RF final metrics:", rf_final_metrics)

            rf_art_final = {
                'model': rf_final,
                'feature_names': X_combined_rf.columns.tolist(),
                'scaler': None,
                'metadata': {
                    'trained_at': datetime.datetime.now().isoformat(),
                    'metrics_final': rf_final_metrics
                }
            }
            joblib.dump(rf_art_final, 'models/rf_final.pkl')
            print("   Saved final RF artifact to models/rf_final.pkl")

            analyze_errors_and_save(y_test_rf, y_pred_rf_final, prefix='reports/rf_error_final')
            print("   RF final error reports saved.")
        except Exception as e:
            print("Warning: Full RF retraining/analysis failed:", e)

    # 9) Podział danych dla XGBoost na train/val/test
    print("-> Splitting XGB datasets (train/val/test)...")
    X = df_xgb.drop(columns=['price'])
    y = df_xgb['price']
    X_train_xgb, X_temp_xgb, y_train_xgb, y_temp_xgb = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val_xgb, X_test_xgb, y_val_xgb, y_test_xgb = train_test_split(X_temp_xgb, y_temp_xgb, test_size=0.5, random_state=42)
    print("   XGB shapes:", X_train_xgb.shape, X_val_xgb.shape, X_test_xgb.shape)

    # 10) Trening XGBoost z kodowaniem Leave-One-Out, ew. tuning Optuna
    print("-> Training XGB with LOO encoder (do_tune=%s) ..." % args.full)
    try:
        artifacts_xgb, (X_test_enc, X_test_scaled, y_test_xgb_local, y_pred_xgb_test) = train_xgb_with_loo(
            X_train_xgb, y_train_xgb,
            X_val_xgb, y_val_xgb,
            X_test_xgb, y_test_xgb,
            do_tune=args.full,
            n_trials=args.n_trials,
            save_plots=args.save_plots
        )
        print("   XGB training completed.")
    except Exception as e:
        print("ERROR training XGB:", e)
        return 9

    # Uzupełnienie metadanych i zapis artefaktów XGB
    print("-> Saving XGB artifacts ...")
    try:
        artifacts_xgb['metadata'].update({
            'train_rows': len(X_train_xgb),
            'val_rows': len(X_val_xgb),
            'test_rows': len(X_test_xgb)
        })
        joblib.dump(artifacts_xgb, 'models/xgb_apartment_artifacts.pkl')
        joblib.dump(artifacts_xgb, 'models/xgb_final.pkl')
        print("   Saved XGB artifacts to models/xgb_apartment_artifacts.pkl and models/xgb_final.pkl")
    except Exception as e:
        print("Warning: Could not save XGB artifacts:", e)

    # Analiza błędów XGB i zapis wykresów
    print("-> Analyzing XGB errors and saving reports...")
    try:
        analyze_errors_and_save(y_test_xgb_local, y_pred_xgb_test, prefix='reports/xgb_error')
        print("   XGB error reports saved.")
    except Exception as e:
        print("Warning: XGB error analysis failed:", e)

    # Analiza SHAP (jeśli wybrano tryb pełny i zapis wykresów)
    if args.full and args.save_plots:
        print("-> Running SHAP analysis and saving plots (full run)...")
        try:
            shap_values = shap_analysis_and_save(
                artifacts_xgb['model'],
                X_test_enc,
                X_test_scaled,
                artifacts_xgb['feature_names'],
                save_prefix='reports/shap_xgb'
            )
            if shap_values is not None:
                mean_abs = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(mean_abs)[-20:][::-1]
                top_feats = [(artifacts_xgb['feature_names'][i], float(mean_abs[i])) for i in top_idx]
                with open('reports/shap_top_features.json', 'w', encoding='utf8') as f:
                    json.dump(top_feats, f, ensure_ascii=False, indent=2)
                print("   SHAP top features saved to reports/shap_top_features.json")
        except Exception as e:
            print("Warning: SHAP analysis failed:", e)

    # Czyszczenie pamięci
    print("-> Cleaning up (gc.collect)...")
    gc.collect()
    print("Pipeline finished.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apartment price modeling pipeline")
    parser.add_argument('--data-dir', type=str, default=str(Path(__file__).parent / 'data'),
                        help='Directory with CSV files')
    parser.add_argument('--full', action='store_true',
                        help='Run full pipeline (EDA, Optuna tuning, SHAP, reports)')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of Optuna trials when --full is used')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to reports/ (instead of showing)')
    args = parser.parse_args()
    rc = main(args)
    sys.exit(rc)
