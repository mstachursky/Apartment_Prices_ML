import joblib
from pathlib import Path
import numpy as np
import datetime
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Opcjonalne importy dla funkcji strojenia hiperparametrów XGBoost za pomocą Optuna i kodowania Leave-One-Out z category_encoders
try:
    import optuna
except Exception:
    optuna = None

try:
    from category_encoders import LeaveOneOutEncoder
except Exception:
    LeaveOneOutEncoder = None

from src.utils import save_figure

def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Trenuje model RandomForestRegressor na danych treningowych.

    Parametry:
        X_train (pd.DataFrame lub np.ndarray): Dane wejściowe do trenowania.
        y_train (pd.Series lub np.ndarray): Wartości docelowe.
        n_estimators (int): Liczba drzew w lesie (domyślnie 100).

    Zwraca:
        RandomForestRegressor: Wytrenowany model.
    """
    # Inicjalizacja modelu RandomForestRegressor z podaną liczbą drzew i innymi parametrami domyślnymi
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    # Trening modelu na danych
    rf.fit(X_train, y_train)
    return rf

def evaluate_model_simple(model, X_test, y_test, scaler=None):
    """
    Ocena modelu na zbiorze testowym. Oblicza RMSE, R2 i MAE.

    Parametry:
        model: Wytrenowany model regresyjny.
        X_test: Dane testowe.
        y_test: Wartości rzeczywiste.
        scaler (opcjonalnie): Obiekt skalera do przekształcenia danych.

    Zwraca:
        tuple: (y_pred, metryki) — predykcje oraz słownik z RMSE, R2, MAE.
    """
    # Skalowanie danych testowych, jeśli podano scaler
    if scaler is not None:
        X_test_in = scaler.transform(X_test)
    else:
        X_test_in = X_test
    # Predykcja na danych testowych
    y_pred = model.predict(X_test_in)
    # Obliczenie metryk jakości
    rmse = float(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    r2 = float(metrics.r2_score(y_test, y_pred))
    mae = float(metrics.mean_absolute_error(y_test, y_pred))
    return y_pred, {'rmse': rmse, 'r2': r2, 'mae': mae}

def tune_xgb_with_optuna(X_train_s, y_train, X_val_s, y_val, n_trials=30, save_prefix='models/xgb_optuna', save_plots=True):
    """
    Przeprowadza strojenie hiperparametrów XGBoost za pomocą Optuna.
    Używa walidacji (X_val_s, y_val) do oceny i early stopping.

    Parametry:
        X_train_s: Przeskalowane dane treningowe.
        y_train: Wartości docelowe treningowe.
        X_val_s: Przeskalowane dane walidacyjne.
        y_val: Wartości docelowe walidacyjne.
        n_trials (int): Liczba prób Optuna.
        save_prefix (str): Prefiks ścieżki do zapisu wyników.
        save_plots (bool): Czy zapisywać wykresy Optuna.

    Zwraca:
        dict: Najlepsze hiperparametry znalezione przez Optuna.
    """
    if optuna is None:
        raise RuntimeError("Optuna not installed. Install optuna to use tuning.")

    def objective(trial):
        # Definicja zakresów hiperparametrów
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        # Inicjalizacja modelu XGBRegressor z wybranymi parametrami
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        # Trening modelu z early stopping na walidacji
        try:
            model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], early_stopping_rounds=30, verbose=False)
        except TypeError:
            try:
                from xgboost import callback
                model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], callbacks=[callback.EarlyStopping(rounds=30)], verbose=False)
            except Exception:
                model.fit(X_train_s, y_train)
        # Predykcja na walidacji i obliczenie RMSE
        preds = model.predict(X_val_s)
        rmse = float(np.sqrt(metrics.mean_squared_error(y_val, preds)))
        return rmse

    # Utworzenie i optymalizacja badania Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Zapis wyników i najlepszych parametrów
    Path(save_prefix + '_files').mkdir(parents=True, exist_ok=True)
    joblib.dump(study, Path(save_prefix + '_files') / 'study.pkl')
    joblib.dump(study.best_params, Path(save_prefix + '_files') / 'best_params.pkl')
    print("Optuna best RMSE:", study.best_value)
    print("Optuna best params:", study.best_params)

    # Opcjonalne wykresy Optuna
    if save_plots:
        try:
            import optuna.visualization.matplotlib as ovm
            fig = ovm.plot_optimization_history(study)
            save_figure(fig.figure, Path('reports') / (save_prefix + '_opt_history.png'))
            fig2 = ovm.plot_param_importances(study)
            save_figure(fig2.figure, Path('reports') / (save_prefix + '_param_importances.png'))
        except Exception as e:
            print("Could not create optuna plots:", e)

    return study.best_params

def train_xgb_with_loo(X_train, y_train, X_val, y_val, X_test, y_test, do_tune=False, n_trials=30, save_plots=False):
    """
    Trenuje model XGBoost z kodowaniem Leave-One-Out dla kolumny 'city'.
    Opcjonalnie stroi hiperparametry za pomocą Optuna.
    Zwraca artefakty modelu oraz dane testowe i predykcje.

    Parametry:
        X_train, X_val, X_test: Dane wejściowe (DataFrame).
        y_train, y_val, y_test: Wartości docelowe.
        do_tune (bool): Czy stroić hiperparametry.
        n_trials (int): Liczba prób Optuna.
        save_plots (bool): Czy zapisywać wykresy Optuna.

    Zwraca:
        tuple: (artefakty modelu, (X_test_enc, X_test_scaled, y_test, y_pred))
    """
    # Sprawdzenie obecności kolumny 'city'
    if 'city' not in X_train.columns:
        raise RuntimeError("Column 'city' missing for XGB LOO pipeline.")
    if LeaveOneOutEncoder is None:
        raise RuntimeError("category_encoders not installed. Install: pip install category_encoders")

    # Inicjalizacja i trening kodera Leave-One-Out na kolumnie 'city'
    encoder = LeaveOneOutEncoder(cols=['city'], sigma=0.05)
    encoder.fit(X_train[['city']], y_train)

    # Kopiowanie danych do kodowania
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()

    # Kodowanie kolumny 'city' w każdym zbiorze
    X_train_enc['city_loo'] = encoder.transform(X_train[['city']])['city']
    X_val_enc['city_loo'] = encoder.transform(X_val[['city']])['city']
    X_test_enc['city_loo'] = encoder.transform(X_test[['city']])['city']

    # Usunięcie oryginalnej kolumny 'city'
    X_train_enc = X_train_enc.drop(columns=['city'])
    X_val_enc = X_val_enc.drop(columns=['city'])
    X_test_enc = X_test_enc.drop(columns=['city'])

    # Konwersja kolumn binarnych do 0/1
    binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    for col in binary_cols:
        if col in X_train_enc.columns:
            if X_train_enc[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                X_train_enc[col] = X_train_enc[col].astype(int)
                X_val_enc[col] = X_val_enc[col].astype(int)
                X_test_enc[col] = X_test_enc[col].astype(int)
            else:
                X_train_enc[col] = (X_train_enc[col].astype(str).str.lower() == 'yes').astype(int)
                X_val_enc[col] = (X_val_enc[col].astype(str).str.lower() == 'yes').astype(int)
                X_test_enc[col] = (X_test_enc[col].astype(str).str.lower() == 'yes').astype(int)

    # Skalowanie cech za pomocą StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train_enc)
    X_train_scaled = scaler.transform(X_train_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    # Strojenie hiperparametrów XGBoost za pomocą Optuna (jeśli wybrano)
    if do_tune:
        best_params = tune_xgb_with_optuna(X_train_scaled, y_train, X_val_scaled, y_val, n_trials=n_trials, save_plots=save_plots)
        best_params['random_state'] = 42
    else:
        best_params = {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1, 'random_state': 42}

    # Połączenie zbiorów treningowych i walidacyjnych do finalnego treningu
    X_combined = np.vstack([X_train_scaled, X_val_scaled])
    y_combined = np.concatenate([y_train, y_val])

    # Trening finalnego modelu XGBRegressor
    xgb_model = XGBRegressor(**best_params, n_jobs=-1)
    try:
        xgb_model.fit(X_combined, y_combined, eval_set=[(X_val_scaled, y_val)], early_stopping_rounds=30, verbose=False)
        print("XGB: trained with early_stopping_rounds (validation).")
    except TypeError:
        try:
            from xgboost import callback
            xgb_model.fit(X_combined, y_combined, eval_set=[(X_val_scaled, y_val)], callbacks=[callback.EarlyStopping(rounds=30)], verbose=False)
            print("XGB: trained with EarlyStopping callback.")
        except Exception:
            print("XGB: training without early stopping.")
            xgb_model.fit(X_combined, y_combined)

    # Predykcja na zbiorze testowym i obliczenie metryk
    y_pred = xgb_model.predict(X_test_scaled)
    rmse = float(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    r2 = float(metrics.r2_score(y_test, y_pred))
    mae = float(metrics.mean_absolute_error(y_test, y_pred))
    print(f"XGB test RMSE: {rmse:.2f}, R2: {r2:.4f}, MAE: {mae:.2f}")

    # Zwrócenie artefaktów modelu oraz danych testowych i predykcji
    artifacts = {
        'model': xgb_model,
        'scaler': scaler,
        'encoder': encoder,
        'feature_names': list(X_train_enc.columns),
        'metadata': {
            'trained_at': datetime.datetime.now().isoformat(),
            'best_params': best_params
        }
    }
    return artifacts, (X_test_enc, X_test_scaled, y_test, y_pred)
