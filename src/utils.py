from pathlib import Path
import datetime
import json
import matplotlib.pyplot as plt

def setup_dirs():
    """
    Tworzy katalogi 'models', 'reports' oraz 'logs' w bieżącym katalogu roboczym, jeśli jeszcze nie istnieją.
    Ułatwia organizację plików projektu.
    """
    # Utwórz katalog 'models' (jeśli nie istnieje)
    Path('models').mkdir(parents=True, exist_ok=True)
    # Utwórz katalog 'reports' (jeśli nie istnieje)
    Path('reports').mkdir(parents=True, exist_ok=True)
    # Utwórz katalog 'logs' (jeśli nie istnieje)
    Path('logs').mkdir(parents=True, exist_ok=True)

def log_prediction(input_data, output_price, intermediate_df=None):
    """
    Zapisuje log predykcji do pliku tekstowego w katalogu 'logs'.
    Dodaje informacje o wejściu, wyjściu oraz (opcjonalnie) użytych cechach.

    Parametry:
        input_data (dict): Dane wejściowe użyte do predykcji.
        output_price (float): Wynik predykcji (cena).
        intermediate_df (pd.DataFrame, opcjonalnie): DataFrame z cechami użytymi do predykcji.
    """
    try:
        # Upewnij się, że katalog 'logs' istnieje
        Path('logs').mkdir(parents=True, exist_ok=True)
        # Otwórz plik logu w trybie dopisywania
        with open(Path('logs') / 'prediction_log.txt', 'a', encoding='utf8') as f:
            # Zapisz znacznik czasu predykcji
            f.write(f"\n--- Prediction at {datetime.datetime.now().isoformat()} ---\n")
            # Zapisz dane wejściowe w formacie JSON
            f.write(f"Input: {json.dumps(input_data, ensure_ascii=False)}\n")
            # Zapisz wynik predykcji
            f.write(f"Output: {float(output_price):.2f}\n")
            # Jeśli podano DataFrame z cechami, zapisz listę kolumn
            if intermediate_df is not None:
                f.write(f"Features used: {intermediate_df.columns.tolist()}\n")
    except Exception as e:
        # W przypadku błędu wyświetl ostrzeżenie
        print("Warning: Could not log prediction:", e)

def save_figure(fig, fname):
    """
    Zapisuje wykres matplotlib do pliku o podanej nazwie.
    Tworzy katalog docelowy, jeśli nie istnieje. Po zapisaniu zamyka wykres.

    Parametry:
        fig (matplotlib.figure.Figure): Obiekt wykresu do zapisania.
        fname (str lub Path): Ścieżka docelowa pliku.
    """
    try:
        # Utwórz katalog docelowy, jeśli nie istnieje
        p = Path(fname)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Zapisz wykres do pliku
        fig.savefig(p, bbox_inches='tight')
        # Zamknij wykres, aby zwolnić zasoby
        plt.close(fig)
    except Exception as e:
        # W przypadku błędu wyświetl ostrzeżenie
        print("Warning: could not save figure", fname, e)
