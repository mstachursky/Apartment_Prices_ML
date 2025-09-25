import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import save_figure
import json
import io
from sklearn import metrics

def evaluate_model(model, X_test, y_test, scaler=None):
    """
    Ocena modelu regresyjnego na zbiorze testowym.
    Przekształca dane testowe (opcjonalnie skalerem), wykonuje predykcję i oblicza metryki RMSE, R2, MAE.

    Parametry:
        model: Wytrenowany model regresyjny.
        X_test: Dane testowe.
        y_test: Wartości rzeczywiste.
        scaler (opcjonalnie): Obiekt skalera do przekształcenia danych.

    Zwraca:
        tuple: (predykcje, słownik metryk)
    """
    # Przekształcenie danych testowych, jeśli podano scaler
    if scaler is not None:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test
    # Predykcja na danych testowych
    preds = model.predict(X_input)
    # Obliczenie metryk jakości
    rmse = float(np.sqrt(metrics.mean_squared_error(y_test, preds)))
    r2 = float(metrics.r2_score(y_test, preds))
    mae = float(metrics.mean_absolute_error(y_test, preds))
    print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}, MAE: {mae:.2f}")
    return preds, {'rmse': rmse, 'r2': r2, 'mae': mae}

def analyze_errors_and_save(y_true, y_pred, prefix='reports/error'):
    """
    Analizuje błędy predykcji i zapisuje wykresy oraz metryki do pliku.
    Oblicza błędy bezwzględne, względne, MAPE, MAE, RMSE, medianę błędu oraz procent trafień poniżej progów.

    Parametry:
        y_true (array): Wartości rzeczywiste.
        y_pred (array): Wartości przewidziane.
        prefix (str): Prefiks ścieżki do zapisu wykresów.

    Zwraca:
        dict: Słownik metryk błędu.
    """
    # Obliczenie błędów predykcji
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / y_true.replace(0, np.nan) * 100
    # Obliczenie metryk błędu
    metrics_dict = {
        'MAE': abs_errors.mean(),
        'RMSE': np.sqrt((errors**2).mean()),
        'MedianAE': np.median(abs_errors),
        'MAPE': rel_errors.mean(),
        'Error < 50k PLN (%)': (abs_errors < 50000).mean() * 100,
        'Error < 100k PLN (%)': (abs_errors < 100000).mean() * 100
    }
    print("\n--- Error metrics ---")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.2f}")

    # Tworzenie wykresów błędów
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Wykres błędów względem wartości rzeczywistych
    axes[0, 0].scatter(y_true, errors, alpha=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Actual price')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].set_title('Errors vs Actual')
    axes[0, 0].grid(True)
    # Histogram błędów
    axes[0, 1].hist(errors, bins=30, alpha=0.7)
    axes[0, 1].set_title('Error distribution')
    # Wykres przewidziane vs rzeczywiste
    axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    axes[1, 0].set_title('Predicted vs Actual')
    # Histogram błędów procentowych
    axes[1, 1].hist(rel_errors.dropna(), bins=30, alpha=0.7)
    axes[1, 1].set_title('Percentage error distribution')
    plt.tight_layout()
    # Zapis wykresu do pliku
    save_figure(fig, Path(f"{prefix}.png"))
    return metrics_dict

def perform_eda_and_save(df, save_prefix='reports/eda'):
    """
    Wykonuje eksploracyjną analizę danych (EDA) i zapisuje podsumowanie oraz wykresy do plików.
    Zapisuje head, info, describe, histogram cen oraz macierz korelacji.

    Parametry:
        df (pd.DataFrame): Dane wejściowe.
        save_prefix (str): Prefiks ścieżki do zapisu raportów.

    Zwraca:
        None
    """
    try:
        # Utworzenie ścieżki do pliku podsumowania
        Path = __import__('pathlib').Path
        out_path = Path('reports') / (save_prefix + '_summary.txt')
        with open(out_path, 'w', encoding='utf8') as f:
            f.write("Head:\n")
            f.write(str(df.head()))
            f.write("\n\nInfo:\n")
            # Zapis df.info() do pliku
            buffer = io.StringIO()
            df.info(buf=buffer)
            f.write(buffer.getvalue())
            f.write("\n\nDescribe:\n")
            f.write(str(df.describe()))
        # Histogram cen
        if 'price' in df.columns:
            fig = plt.figure(figsize=(8, 4))
            plt.hist(df['price'].dropna(), bins=50, alpha=0.7)
            plt.title('Price distribution')
            save_figure(fig, Path('reports') / (save_prefix + '_price_hist.png'))
        # Macierz korelacji dla kolumn numerycznych
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            try:
                import seaborn as sns
                fig = plt.figure(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
                plt.title('Correlation matrix')
                save_figure(fig, Path('reports') / (save_prefix + '_corr.png'))
            except Exception as e:
                print("Seaborn not available or heatmap failed:", e)
        print("EDA saved to reports/")
    except Exception as e:
        print("EDA failed:", e)

def shap_analysis_and_save(model, X_test_raw, X_test_scaled, feature_names, save_prefix='reports/shap'):
    """
    Przeprowadza analizę SHAP dla modelu drzewiastego i zapisuje wykresy podsumowujące do plików.
    Tworzy wykresy summary oraz bar plot dla ważności cech.

    Parametry:
        model: Wytrenowany model (np. XGBRegressor).
        X_test_raw: Dane testowe w oryginalnej postaci.
        X_test_scaled: Dane testowe przeskalowane.
        feature_names (list): Lista nazw cech.
        save_prefix (str): Prefiks ścieżki do zapisu wykresów.

    Zwraca:
        shap_values lub None
    """
    try:
        import shap
    except Exception:
        print("SHAP not installed, skipping SHAP analysis.")
        return None

    try:
        # Inicjalizacja explainer'a SHAP dla modelu drzewiastego
        explainer = shap.TreeExplainer(model)
        # Obliczenie wartości SHAP dla danych testowych
        shap_values = explainer.shap_values(X_test_scaled)
        # Wykres summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test_raw, feature_names=feature_names, show=False)
        save_figure(plt.gcf(), Path(f"{save_prefix}_summary.png"))
        plt.close('all')
        # Wykres bar plot dla ważności cech
        plt.figure()
        shap.summary_plot(shap_values, X_test_raw, feature_names=feature_names, plot_type="bar", show=False)
        save_figure(plt.gcf(), Path(f"{save_prefix}_bar.png"))
        plt.close('all')
        print("SHAP plots saved.")
        return shap_values
    except Exception as e:
        print("SHAP analysis failed:", e)
        return None
