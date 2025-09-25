import joblib
import pandas as pd
from pathlib import Path
from src.utils import log_prediction

def predict_single(features_dict, artifact_path='models/xgb_final.pkl', mode='xgb'):
    """
    Przewiduje cenę dla pojedynczej próbki na podstawie słownika cech.
    Wczytuje artefakty modelu (model, scaler, encoder, feature_names) z pliku.
    Przygotowuje cechy zgodnie z trybem ('xgb' lub 'rf'):
      - 'xgb': kodowanie Leave-One-Out dla miasta, binarne cechy jako 0/1.
      - 'rf': one-hot encoding dla miasta i cech binarnych.
    Uzupełnia brakujące cechy zerami, dopasowuje kolejność kolumn.
    Przekształca dane skalarem (jeśli dostępny), wykonuje predykcję i zapisuje log.

    Parametry:
        features_dict (dict): Słownik cech wejściowych.
        artifact_path (str): Ścieżka do pliku z artefaktami modelu.
        mode (str): 'xgb' lub 'rf' — sposób przygotowania cech.

    Zwraca:
        float: Przewidziana cena.
    """
    # Wczytaj artefakty modelu z pliku
    art = joblib.load(artifact_path)
    model = art['model']
    scaler = art.get('scaler', None)
    encoder = art.get('encoder', None)
    feature_names = art['feature_names']

    # Utwórz DataFrame z pojedynczej próbki
    df = pd.DataFrame([features_dict])
    # Standaryzuj nazwę miasta
    if 'city' in df.columns:
        df['city'] = df['city'].astype(str).str.lower().str.strip()

    binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']

    if mode == 'xgb':
        # Kodowanie Leave-One-Out dla miasta
        if encoder is not None:
            try:
                df['city_loo'] = encoder.transform(df[['city']])['city']
            except Exception:
                # Fallback: użyj mapowania z artefaktów lub średniej globalnej
                loo_map = art.get('city_loo_map', None)
                global_mean = art.get('global_loo_mean', None)
                city = df['city'].iloc[0] if 'city' in df.columns else None
                if loo_map is not None and city in loo_map:
                    df['city_loo'] = loo_map[city]
                else:
                    df['city_loo'] = (global_mean if global_mean is not None else 0.0)
        else:
            # Brak encoder — fallback do mapowania z artefaktów
            loo_map = art.get('city_loo_map', None)
            global_mean = art.get('global_loo_mean', None)
            city = df['city'].iloc[0] if 'city' in df.columns else None
            df['city_loo'] = loo_map.get(city, global_mean) if loo_map is not None else (global_mean or 0.0)

        # Usuń oryginalną kolumnę miasta
        if 'city' in df.columns:
            df = df.drop(columns=['city'])

        # Zamień wartości binarne na 0/1
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map({'yes':1,'no':0,'tak':1,'nie':0,'1':1,'0':0}).fillna(0).astype(int)

    else:
        # RandomForest: one-hot encoding dla miasta i cech binarnych
        if 'city' in df.columns:
            city_val = df['city'].iloc[0]
            for col in feature_names:
                if col.startswith('city_'):
                    city_name = col[len('city_'):]
                    df[col] = int(city_val == city_name)
            df = df.drop(columns=['city'])

        # Rozwijanie binarnych dummy columns
        for b in binary_cols:
            yes_name = f"{b}_yes"
            no_name = f"{b}_no"
            if yes_name in feature_names or no_name in feature_names:
                val = df[b].iloc[0] if b in df.columns else None
                mapped = None
                if val is None:
                    mapped = None
                else:
                    s = str(val).strip().lower()
                    if s in ['yes','tak','1','1.0','true','t','y']:
                        mapped = 'yes'
                    elif s in ['no','nie','0','0.0','false','f','n']:
                        mapped = 'no'
                if yes_name in feature_names:
                    df[yes_name] = 1 if mapped == 'yes' else 0
                if no_name in feature_names:
                    df[no_name] = 1 if mapped == 'no' else 0
        # Usuń oryginalne kolumny binarne
        for b in binary_cols:
            if b in df.columns:
                df = df.drop(columns=[b])

    # Uzupełnij brakujące cechy zerami i ustaw kolejność kolumn
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Przekształć dane skalarem (jeśli dostępny)
    X_input = scaler.transform(df) if scaler is not None else df.values
    try:
        # Predykcja modelu
        price = float(model.predict(X_input)[0])
        # Zapisz log predykcji
        log_prediction(features_dict, price, intermediate_df=df)
        return price
    except Exception as e:
        raise RuntimeError("Prediction failed: " + str(e))
