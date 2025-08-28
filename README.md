# Credit Risk Modeling – Streamlit App

Diese Anwendung rekonstruiert ein **Credit Risk Modeling** Projekt mit Streamlit.  
Features:
- Laden von `german_credit_data.csv`
- Explorative Datenanalyse (EDA) mit Plotly
- Modelltraining (XGBoost + GridSearchCV, LabelEncoder für Kategorien)
- Vorhersage einzelner Datensätze mit interaktiver Eingabe

---

## 🚀 Installation

1. Repository klonen:
   ```bash
   git clone git@github.com:Alexbalzer/Credit-Risk-Modeling-Streamlit-App.git
   cd Credit-Risk-Modeling-Streamlit-App


2. Virtuelle Umgebung erstellen und Abhängigkeiten installieren:

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt

2. CSV-Datei bereitstellen:
    Lege die Datei ´german_credit_data.csv´ in das Projektverzeichnis.

## 📂 Projektstruktur
Credit-Risk-Modeling-Streamlit-App/
│── app.py
│── model_utils.py
│── requirements.txt
│── README.md
│── .gitignore
└── german_credit_data.csv   # (nicht im Repo enthalten)

## 📂 Hinweise

- Beim ersten Training werden ein XGBoost-Modell und LabelEncoder als .pkl gespeichert.

- Trainierte Artefakte werden nicht versioniert (siehe .gitignore).