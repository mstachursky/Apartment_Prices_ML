import tkinter as tk
from tkinter import messagebox, ttk
from contextlib import suppress
from src.predict import predict_single

# ========================
# Konfiguracja
# ========================
MODEL_PATH = "models/xgb_final.pkl"

DEFAULT_VALUES = {
    "city": "gdynia",
    "buildYear": "2012",
    "squareMeters": "50.0",
    "rooms": "2",
    "floor": "1",
    "floorCount": "4",
    "hasParkingSpace": "yes",
    "hasBalcony": "yes",
    "hasElevator": "yes",
    "hasSecurity": "yes",
    "hasStorageRoom": "yes",
    "poiCount": "3",
    "centreDistance": "5.0",
    "clinicDistance": "2.0",
    "restaurantDistance": "1.0",
    "collegeDistance": "4.0",
    "pharmacyDistance": "1.0",
    "postOfficeDistance": "3.0",
    "schoolDistance": "1.0",
    "kindergartenDistance": "1.0"
}

FIELDS = {
    "city": "Miasto",
    "buildYear": "Rok budowy",
    "squareMeters": "Powierzchnia (m²)",
    "rooms": "Liczba pokoi",
    "floor": "Piętro",
    "floorCount": "Liczba pięter w budynku",
    "hasParkingSpace": "Parking",
    "hasBalcony": "Balkon",
    "hasElevator": "Winda",
    "hasSecurity": "Ochrona",
    "hasStorageRoom": "Komórka lokatorska",
    "poiCount": "Liczba punktów POI w okolicy (500 m)",
    "centreDistance": "Odległość od centrum (km)",
    "clinicDistance": "Odległość od szpitala (km)",
    "restaurantDistance": "Odległość od punktu gastro (km)",
    "collegeDistance": "Odległość od uczelni wyższej (km)",
    "pharmacyDistance": "Odległość od apteki (km)",
    "postOfficeDistance": "Odległość od poczty (km)",
    "schoolDistance": "Odległość od szkoły (km)",
    "kindergartenDistance": "Odległość od przedszkola (km)"

}

CITY_OPTIONS = [
    "bialystok", "bydgoszcz", "czestochowa", "gdansk", "gdynia", "katowice",
    "krakow", "lodz", "lublin", "poznan", "radom", "rzeszow",
    "szczecin", "warszawa", "wroclaw"
]

YES_NO_OPTIONS = ["tak", "nie"]

FLOAT_FIELDS = {"squareMeters", "centreDistance", "clinicDistance", "restaurantDistance", "collegeDistance", "pharmacyDistance", "postOfficeDistance", "schoolDistance", "kindergartenDistance"}
INT_FIELDS = {"buildYear", "rooms", "floor", "floorCount", "poiCount"}

def validate_number(new_value, field_type):
    if new_value == "":
        return True
    if field_type == "int":
        return new_value.isdigit()
    elif field_type == "float":
        with suppress(ValueError):
            val = float(new_value)
            parts = new_value.split(".")
            if len(parts) == 2 and len(parts[1]) > 1:
                return False
            return True
    return False

def predict_price():
    try:
        features = {}
        for key, widget in entries.items():
            if isinstance(widget, tk.StringVar):
                value = widget.get().strip().lower()
                if key == "city":
                    if not value:
                        raise ValueError("Miasto nie może być puste.")
                    features[key] = value
                else:
                    features[key] = "yes" if value == "tak" else "no"
            else:
                value = widget.get().strip()
                if key in INT_FIELDS:
                    features[key] = int(value) if value else 0
                elif key in FLOAT_FIELDS:
                    features[key] = round(float(value), 1) if value else 0.0
                else:
                    features[key] = value or ""
        price = predict_single(features, artifact_path=MODEL_PATH, mode="xgb")
        messagebox.showinfo("Predykcja ceny", f"Przewidywana cena mieszkania: {price:,.2f} PLN")
    except Exception as e:
        messagebox.showerror("Błąd", f"Coś poszło nie tak: {e}")

def reset_fields():
    for key, widget in entries.items():
        default = DEFAULT_VALUES.get(key, "")
        if isinstance(widget, tk.StringVar):
            if key in ["hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]:
                widget.set("nie")
            elif key == "city":
                widget.set("gdynia")
        else:
            widget.delete(0, tk.END)
            widget.insert(0, default)

def clear_manual_fields():
    bool_keys = {"hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"}
    for key, widget in entries.items():
        if isinstance(widget, tk.StringVar):
            # ustawiamy tylko radiobuttony, pomijamy miasto
            if key in bool_keys:
                widget.set("nie")
            else:
                continue
        else:
            # spinbox / entry -> wyczyść
            try:
                widget.delete(0, tk.END)
            except Exception:
                pass

# ========================
# GUI - styl "Apple"
# ========================
root = tk.Tk()
root.title("Predykcja cen mieszkań")

# Stylizacja inspirowana macOS
BG_COLOR = "#f8f9fa"  # Bardziej neutralny, lekko szary odcień
LABEL_COLOR = "#212529"  # Głębszy, bardziej elegancki prawie-czarny
ENTRY_BG = "#ffffff"    # Czysta biel
BUTTON_BG = "#495057"   # Elegancki ciemny szary zamiast standardowego niebieskiego
BUTTON_ACTIVE_BG = "#343a40"  # Ciemniejszy odcień dla stanu aktywnego
BUTTON_FG = "#ffffff"   # Biały tekst
RADIO_BG = "#f8f9fa"    # Dopasowany do nowego BG_COLOR
FONT = ("SF Pro Text", 12)  # Nowoczesna czcionka Apple
FONT_BOLD = ("SF Pro Display", 12, "bold")  # Display wariant dla pogrubienia

root.configure(bg=BG_COLOR)
style = ttk.Style(root)
style.theme_use("clam")

style.configure("TLabel", background=BG_COLOR, foreground=LABEL_COLOR, font=FONT)
style.configure("TButton", font=FONT_BOLD, padding=10, background=BUTTON_BG, foreground=BUTTON_FG)
style.map("TButton",
    background=[("active", BUTTON_ACTIVE_BG), ("!active", BUTTON_BG)],
    foreground=[("active", BUTTON_FG), ("!active", BUTTON_FG)]
)
style.configure("TEntry", font=FONT, fieldbackground=ENTRY_BG)
style.configure("TCombobox", font=FONT, fieldbackground=ENTRY_BG)
style.configure("TRadiobutton", background=RADIO_BG, font=FONT)

main_frame = ttk.Frame(root, padding=16, style="TFrame")  # zmniejszono padding ramki
main_frame.grid(row=0, column=0, sticky="nsew")
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

entries = {}

for i, (key, label) in enumerate(FIELDS.items()):
    # zmniejszono pady i padx dla każdego wiersza
    ttk.Label(main_frame, text=label).grid(row=i, column=0, padx=6, pady=5, sticky="w")
    if key == "city":
        var = tk.StringVar(value=DEFAULT_VALUES["city"])
        combo = ttk.Combobox(main_frame, textvariable=var, values=CITY_OPTIONS, state="readonly", width=24)
        combo.grid(row=i, column=1, padx=6, pady=5, sticky="ew")
        combo.configure(font=FONT)
        entries[key] = var
    elif key in ["hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]:
        var = tk.StringVar(value="nie")
        radio_frame = ttk.Frame(main_frame, style="TFrame")
        radio_frame.grid(row=i, column=1, padx=6, pady=5, sticky="ew")
        ttk.Radiobutton(radio_frame, text="Tak", variable=var, value="tak").pack(side="left", padx=5)
        ttk.Radiobutton(radio_frame, text="Nie", variable=var, value="nie").pack(side="left", padx=5)
        entries[key] = var
    else:
        field_type = "float" if key in FLOAT_FIELDS else "int"
        vcmd = (root.register(validate_number), "%P", field_type)
        if field_type == "int":
            spin = ttk.Spinbox(main_frame, from_=0, to=9999, width=22, validate="key", validatecommand=vcmd)
            spin.grid(row=i, column=1, padx=6, pady=5, sticky="ew")
            spin.delete(0, tk.END)
            spin.insert(0, DEFAULT_VALUES[key])
            spin.configure(font=FONT)
            entries[key] = spin
        else:
            spin = ttk.Spinbox(main_frame, from_=0, to=9999, increment=1.0, format="%.1f", width=22, validate="key", validatecommand=vcmd)
            spin.grid(row=i, column=1, padx=6, pady=5, sticky="ew")
            spin.delete(0, tk.END)
            spin.insert(0, DEFAULT_VALUES[key])
            spin.configure(font=FONT)
            entries[key] = spin

main_frame.grid_columnconfigure(1, weight=1)

btn_frame = ttk.Frame(main_frame, style="TFrame")
btn_frame.grid(row=len(FIELDS), column=0, columnspan=2, pady=12)  # zmniejszono pady

btn_predict = ttk.Button(btn_frame, text="Zrób czary-mary", command=predict_price)
btn_predict.pack(side="left", padx=10)
btn_predict.configure(style="TButton")

btn_reset = ttk.Button(btn_frame, text="Przywróć domyślne", command=reset_fields)
btn_reset.pack(side="left", padx=10)
btn_reset.configure(style="TButton")

btn_clear = ttk.Button(btn_frame, text="Wyczyść", command=clear_manual_fields)
btn_clear.pack(side="left", padx=10)
btn_clear.configure(style="TButton")

root.mainloop()