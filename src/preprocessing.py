import pandas as pd
import numpy as np

def remove_percentile_outliers_per_city(df, target_col='price', lower_p=0.01, upper_p=0.95):
    """
    Usuwa obserwacje, których wartość w kolumnie target_col znajduje się poza zakresem percentyli lower_p i upper_p
    w każdej grupie 'city'. Pozwala to na usunięcie skrajnych wartości osobno dla każdego miasta.
    
    Parametry:
        df (pd.DataFrame): Dane wejściowe.
        target_col (str): Kolumna, dla której liczone są percentyle (domyślnie 'price').
        lower_p (float): Dolny percentyl (domyślnie 0.01).
        upper_p (float): Górny percentyl (domyślnie 0.95).
    
    Zwraca:
        pd.DataFrame: Odfiltrowany DataFrame bez skrajnych wartości w każdej grupie 'city'.
    """
    # Sprawdź, czy istnieje kolumna 'city'
    if 'city' not in df.columns:
        return df.copy().reset_index(drop=True)
    frames = []
    # Iteruj po każdej grupie (miasto)
    for city, group in df.groupby('city'):
        # Wyznacz dolny i górny percentyl dla target_col w danym mieście
        p_low = group[target_col].quantile(lower_p)
        p_high = group[target_col].quantile(upper_p)
        # Wybierz obserwacje w zakresie percentyli
        filtered = group[(group[target_col] >= p_low) & (group[target_col] <= p_high)].copy()
        frames.append(filtered)
    # Połącz wszystkie przefiltrowane grupy w jeden DataFrame
    res = pd.concat(frames).reset_index(drop=True)
    print(f"After removing percentile outliers per city: {res.shape}")
    return res

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wykonuje podstawowe czyszczenie danych:
      - Usuwa zbędne kolumny.
      - Standaryzuje nazwy miast.
      - Konwertuje wybrane kolumny na typ numeryczny.
      - Wypełnia brakujące wartości medianą.
      - Ujednolica wartości binarne do 'yes'/'no'.
    
    Parametry:
        df (pd.DataFrame): Dane wejściowe.
    
    Zwraca:
        pd.DataFrame: Oczyszczony DataFrame.
    """
    df = df.copy()
    # Usuń zbędne kolumny, jeśli istnieją
    cols_to_drop = ['id', 'buildingMaterial', 'condition', 'type', 'ownership', 'latitude', 'longitude', 'source_file']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Standaryzuj nazwy miast (małe litery, bez spacji)
    if 'city' in df.columns:
        df['city'] = df['city'].astype(str).str.lower().str.strip()

    # Konwertuj wybrane kolumny na typ numeryczny
    numeric_cols = ['buildYear', 'squareMeters', 'rooms', 'poiCount']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Wypełnij brakujące wartości w 'floorCount' medianą
    if 'floorCount' in df.columns:
        df['floorCount'] = df['floorCount'].fillna(df['floorCount'].median())
    # Wypełnij brakujące wartości w 'floor' połową liczby pięter
    if 'floor' in df.columns and 'floorCount' in df.columns:
        df['floor'] = df['floor'].fillna((df['floorCount'] // 2).astype(float))

    # Wypełnij brakujące wartości w kolumnach odległości medianą
    poi_columns = ['schoolDistance', 'clinicDistance', 'postOfficeDistance',
                   'kindergartenDistance', 'restaurantDistance', 'collegeDistance',
                   'pharmacyDistance']
    for col in poi_columns:
        if col in df.columns:
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)

    # Ujednolicenie wartości binarnych do 'yes'/'no'
    binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    map_vals = {
        'true':'yes','false':'no','1':'yes','0':'no',
        'nan':'no','none':'no','tak':'yes','nie':'no','yes':'yes','no':'no'
    }
    for col in binary_cols:
        if col in df.columns:
            # Jeśli kolumna jest numeryczna, zamień 1/0 na 'yes'/'no'
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                df[col] = np.where(df[col].isin([0,1]), df[col], np.nan)
                df[col] = df[col].map({1:'yes', 0:'no'}).fillna('no')
            else:
                # Jeśli kolumna jest tekstowa, zmapuj wartości na 'yes'/'no'
                df[col] = df[col].astype(str).str.lower().str.strip().map(map_vals).fillna('no')
    return df

def impute_numeric_per_city_with_median(df, numeric_cols):
    """
    Wypełnia brakujące wartości w podanych kolumnach numerycznych medianą z danej grupy 'city'.
    Jeśli brak kolumny 'city', zwraca kopię danych bez zmian.
    
    Parametry:
        df (pd.DataFrame): Dane wejściowe.
        numeric_cols (list): Lista kolumn numerycznych do imputacji.
    
    Zwraca:
        pd.DataFrame: DataFrame z uzupełnionymi wartościami.
    """
    # Sprawdź, czy istnieje kolumna 'city'
    if 'city' not in df.columns:
        return df.copy()
    frames = []
    # Iteruj po każdej grupie (miasto)
    for city, group in df.groupby('city'):
        g = group.copy()
        # Wypełnij NaN medianą dla każdej kolumny numerycznej
        for col in numeric_cols:
            if col in g.columns:
                g[col] = g[col].fillna(g[col].median())
        frames.append(g)
    # Połącz wszystkie grupy w jeden DataFrame
    return pd.concat(frames).reset_index(drop=True)

def prepare_datasets(df: pd.DataFrame):
    """
    Przygotowuje dwa warianty danych:
      - df_rf: Dane z one-hot encoding dla 'city' i cech binarnych (do RandomForest).
      - df_xgb: 'city' jako kategoria tekstowa, cechy binarne jako 0/1 (do XGBoost i LOO encoding).
    
    Parametry:
        df (pd.DataFrame): Dane wejściowe.
    
    Zwraca:
        tuple: (df_rf, df_xgb) — dwa DataFrame'y gotowe do trenowania różnych modeli.
    """
    df = df.copy()
    # Lista kolumn numerycznych do imputacji
    numeric_cols = ['buildYear', 'squareMeters', 'rooms', 'poiCount']
    # Wypełnij brakujące wartości medianą w każdej grupie 'city'
    df = impute_numeric_per_city_with_median(df, numeric_cols)

    # Przygotowanie wariantu dla RandomForest: one-hot encoding
    df_rf = df.copy()
    binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    cols_for_dummies = []
    if 'city' in df_rf.columns:
        cols_for_dummies.append('city')
    cols_for_dummies += [c for c in binary_cols if c in df_rf.columns]
    # Jeśli są kolumny do zakodowania, wykonaj one-hot encoding
    if cols_for_dummies:
        df_rf = pd.get_dummies(df_rf, columns=cols_for_dummies, drop_first=False, dtype=int)

    # Przygotowanie wariantu dla XGBoost: 'city' jako tekst, binarne jako 0/1
    df_xgb = df.copy()
    for col in binary_cols:
        if col in df_xgb.columns:
            df_xgb[col] = (df_xgb[col] == 'yes').astype(int)

    return df_rf, df_xgb
