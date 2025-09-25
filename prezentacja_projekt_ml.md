# 📊 Omówienie projektu: Predykcja cen mieszkań z użyciem ML

## 🎯 Cel projektu



Cały kod jest częścią naszego projektu końcowego z ML, w którym staramy przewiedzieć ceny mieszkań na podstawie szeregu parametrów, takich jak miasto, metraż, liczba pokoi, piętro, liczba pięter w budynku, rok budowy, odległość od centrum miasta, odległość od poszczególnych POI, oraz udogodnień, takich jak winda, garaż, miejsce parkingowe, ochrona, balkon i komórka lokatorska.

Ponadto chcielibyśmy znaleźć odpowiedzi na następujących pytania:
1. Jak rozkładają się ceny mieszkań w poszczególnych miastach.
2. Jakie parametry mieszkań i ich udogodnienia najbardziej wpływają na cenę.

Celem było zbudowanie modelu uczenia maszynowego, który na podstawie
danych o mieszkaniu (lokalizacja, metraż, liczba pokoi, piętro,
udogodnienia, odległość od centrum, itp.) potrafi **oszacować cenę
mieszkania**.

## 📂 Dane wejściowe

Wykorzystywany zbiór danych pochodzi z serwisu [Kaggle.com](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland/data "Apartment Prices in Poland") i zawiera ceny mieszkań z 15 największych miast w Polsce. Rzeczone dane były zbierane co miesiąc, od sierpnia 2023 do czerwca 2024.

Po pobraniu paczki spakowanych danych wyłaniają się przed nami pliki o następującej strukturze:
* apartments_pl_YYYY_MM.csv
* apartments_rent_pl_YYYY_MM.csv

Na potrzeby naszej analizy będziemy korzystać tylko ze zbioru **apartments_pl_YYYY_MM.csv**, gdyż będziemy analizować mieszkania wystawione na sprzedaż, a nie na wynajem.

W datasecie znajduje się 195568 obserwacji.

Dane pochodzą z ogłoszeń, a nie z aktów notarialnych, w związku z czym mogą zawierać ceny ofertowe, a nie transakcyjne. Ponadto oferty pochodzą z kolejnych miesięcy, więc część z nich może się powtarzać.

-   Każdy rekord opisuje jedno mieszkanie i zawiera m.in.:
    -   **city** - miasto,
    -   **buildYear** - rok budowy,
    -   **squareMeters** - powierzchnia,
    -   **rooms** - liczba pokoi,
    -   **floor**, **floorCount** - piętro i całkowita liczba pięter w budynku
    -   **hasBalcony**, **hasElevator**, **hasParkingSpace**, **hasSecurity**, **hasStorage** - dostępność balkonu, windy, miejsca parkingowego, ochrony, komórki lokatorskiej,
    -   **distance features** - odległość od centrum, uczelni, kliniki,
        restauracji, apteki, przedszkola,
    -   **price** - cena mieszkania (zmienna docelowa).

## 🔄 Pipeline - przetwarzanie danych

1.  **Ładowanie i łączenie plików CSV**
    -   usunięcie duplikatów (po `id`).
2.  **Czyszczenie danych**
    -   ujednolicenie nazw miast,
    -   konwersja kolumn binarnych do `yes/no`,
    -   uzupełnianie braków medianą (np. piętra, liczba pięter,
        odległości).
3.  **Usuwanie wartości odstających**
    -   outliery w cenach eliminowane metodą IQR osobno dla każdego
        miasta.
4.  **Tworzenie dwóch wersji datasetu:**
    -   dla Random Forest → one-hot encoding,
    -   dla XGBoost → Leave-One-Out Encoding dla miasta + binaria jako
        0/1.
5.  **Podział danych**
    -   Train / Validation / Test (60% / 20% / 20%).

## 🤖 Modele

-   **Random Forest Regressor**
    -   szybki baseline,
    -   działa na zakodowanych zmiennych kategorycznych (one-hot).
-   **XGBoost Regressor**
    -   model drzew gradientowych,
    -   lepiej radzi sobie z nieliniowościami,
    -   możliwość strojenia hiperparametrów z **Optuna**.

## 📈 Ewaluacja modeli

-   Wskaźniki:
    -   **MAE** (średni błąd bezwzględny),
    -   **RMSE** (pierwiastek błędu średniokwadratowego),
    -   **R²** (współczynnik determinacji),
    -   dodatkowo MAPE (%) i odsetek prognoz z błędem \< 50k i \< 100k
        PLN.

### Przykładowe wyniki (dla 250 próbek testowych):

-   **MAE:** 65 225 PLN
-   **RMSE:** 95 016 PLN
-   **MedianAE:** 44 232 PLN
-   **MAPE:** 9,4 %
-   **Error \< 50k PLN:** 54,5 %
-   **Error \< 100k PLN:** 79,6 %

👉 Interpretacja:
Model w typowych przypadkach przewiduje cenę z dokładnością ok.
**±9,4**. Dla mieszkania za 600 tys. PLN to błąd **±56 tys. PLN**.  W **54,5%** przypadków błąd prognozy był mniejszy niż 50 tys. PLN. Największe błędy dotyczą nietypowych mieszkań (np. bardzo luksusowych). Prawie **80% prognoz** mieści się w błędzie do 100 tys. PLN.

## 🧠 Interpretacja modeli

-   W projekcie wykorzystano **SHAP** do oceny ważności cech.
-   Najbardziej wpływowe cechy to:
    -   miasto,
    -   powierzchnia mieszkania,
    -   rok budowy,
    -   odległość od centrum,
    -   liczba pokoi,
    -   liczba punktów POI w obrębie 500 metrów,
    -   obecność windy
    -   odległość od szpitala
    -   odległość od restauracji
-   W folderze `reports/` zapisywane są:
    -   wykresy błędów (Predicted vs Actual, histogramy błędów),
    -   wykresy SHAP (ważność cech, rozkład wpływów).

## 🖥️ Interfejs użytkownika (GUI)

-   Zaimplementowano prosty **interfejs w Tkinterze**
    (`predict_ui.py`).
-   Użytkownik wybiera miasto, podaje parametry mieszkania (metraż,
    pokoje, piętro, odległości, udogodnienia).
-   Po kliknięciu **„Zrób czary-mary"** aplikacja korzysta z
    wytrenowanego modelu (`xgb_final.pkl`) i zwraca przewidywaną cenę.
-   Możliwość resetu pól do wartości domyślnych.

## 📦 Struktura projektu

    ML_Project_Final/
    │
    ├── src/                 # moduły logiki projektu
    │   ├── data_loader.py   # wczytywanie i czyszczenie danych
    │   ├── preprocessing.py # transformacje, imputacje
    │   ├── modeling.py      # trenowanie RF i XGB
    │   ├── evaluate.py      # metryki, wykresy, SHAP
    │   ├── utils.py         # logowanie, zapisywanie plików
    │   └── predict.py       # predykcja pojedynczego przykładu
    │
    ├── models/              # zapisane modele (.pkl)
    ├── reports/             # raporty, wykresy, logi
    ├── main.py              # główny pipeline (train, eval, shap)
    └── predict_ui.py        # GUI do predykcji

## ▶️ Jak uruchomić projekt i wykonywać predykcję

1. Przygotowanie środowiska (Windows PowerShell):
   - Utwórz i aktywuj virtualenv:
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Zainstaluj zależności:
     ```
     pip install -r requirements.txt
     ```
     
2. Uruchamianie pipeline (z katalogu projektu):
   - Szybkie uruchomienie (tylko wczytywanie/oczyszczanie/krótki trening):
     ```
     python main.py --data-dir data
     ```
   - Pełny przebieg (EDA, strojenie Optuna, SHAP, zapisy wykresów):
     ```
     python main.py --data-dir data --full --n-trials 50 --save-plots
     ```
   - Parametry:
     - --data-dir <folder> — folder z CSV,
     - --full — uruchamia pełny workflow,
     - --n-trials <int> — liczba prób Optuna (używana gdy --full),
     - --save-plots — zapisuje wykresy do reports/.

3. Predykcja:
   - GUI (banalne w obsłudze):
     ```
     python predict_ui.py
     ```
     - Po otwarciu okna wprowadź dane i kliknij „Zrób czary‑mary”.
     
4. Uwagi praktyczne:
   - Uruchamiaj skrypty z katalogu głównego projektu (aby importy działały).
   - Jeśli model/artefakty nie istnieją — uruchom `main.py --full` lub wczytaj wytrenowany plik do `models/`.



## ✅ Wnioski

-   Projekt pokazuje praktyczne zastosowanie ML w branży nieruchomości.
-   Zbudowany pipeline działa **end-to-end**: od wczytania surowych
    danych, przez czyszczenie, uczenie modeli, do raportów i GUI.
-   **Model XGBoost** daje trafne prognozy - błąd **~9,4%** w cenach
    mieszkań.
-   Gotowy GUI umożliwia wykorzystanie modelu nawet osobom bez wiedzy o
    ML.
-   Model wymaga douczenia na aktualnych danych, aby odzwierciedlał obecne warunki rynkowe.
