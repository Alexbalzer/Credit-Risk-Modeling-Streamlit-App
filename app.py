import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_utils import (
    load_or_train_model,
    get_feature_config,
    encode_features_for_inference,
    MODEL_PATH,
)

st.set_page_config(page_title="Credit Risk Modeling",page_icon="💳", layout="wide")

st.title("Credit Risk Modeling – Streamlit App")
st.caption("Rekonstruiert aus einer Notebook-Lösung: EDA, Training und Vorhersage")

# --- Sidebar: Datenquelle ---
st.sidebar.header("Daten")
csv_file = st.sidebar.text_input(
    "Pfad zur CSV (german_credit_data.csv)",
    value="german_credit_data.csv"
)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

# Try to read data
df = None
data_error = None
try:
    df = load_data(csv_file)
except Exception as e:
    data_error = str(e)

if data_error:
    st.error(f"Konnte CSV nicht laden: {data_error}")
    st.stop()

st.success(f"CSV geladen: {csv_file}  |  Zeilen: {len(df):,}  Spalten: {df.shape[1]}")

# --- Tabs ---
tab_overview, tab_eda, tab_train, tab_predict = st.tabs(
    ["Überblick", "EDA", "Training", "Vorhersage"]
)

with tab_overview:
    st.subheader("Datenbeispiel")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("**Zielvariable:**", "`Risk` (z. B. 'good'/'bad')")
    st.write("**Typische Merkmale** (abhängig von CSV):", ", ".join([c for c in df.columns if c != "Risk"]))

    st.markdown("""
    **Hinweis:** Die App verwendet *LabelEncoder* pro kategorialer Spalte (wie im Notebook),
    trainiert ein XGBoost‑Modell (GridSearchCV) und speichert die Encoder + das Modell als `.pkl`.
    """)

with tab_eda:
    st.subheader("Verteilung der Zielvariable")
    if "Risk" in df.columns:
        fig = px.histogram(df, x="Risk", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Spalte 'Risk' wurde nicht gefunden.")

    st.divider()
    st.subheader("Boxplots ausgewählter numerischer Merkmale nach 'Risk'")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "Risk"]
    if len(num_cols) >= 1 and "Risk" in df.columns:
        cols = st.multiselect("Numerische Spalten wählen", num_cols, default=num_cols[:3])
        if cols:
            rows = int(np.ceil(len(cols) / 3))
            fig2 = make_subplots(rows=rows, cols=3, subplot_titles=cols)
            r = c = 1
            for col in cols:
                fig2.add_trace(
                    go.Box(
                        x=df["Risk"],
                        y=df[col],
                        boxpoints="outliers",
                        marker=dict(color="lightblue"),
                        line=dict(color="black"),
                        name=col
                    ),
                    row=r, col=c
                )
                c += 1
                if c == 4:
                    c = 1
                    r += 1

            fig2.update_xaxes(showgrid=False)
            fig2.update_yaxes(showgrid=True)
            fig2.update_layout(height=300*rows, showlegend=False, title="Boxplots")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Bitte Spalten auswählen.")
    else:
        st.warning("Keine numerischen Spalten gefunden oder 'Risk' fehlt.")

with tab_train:
    st.subheader("Modelltraining")
    st.write("Das Training nutzt einen `LabelEncoder` pro Kategorikspalte und XGBoost mit GridSearchCV.")
    with st.spinner("Training läuft (oder Laden, falls bereits vorhanden)…"):
        model, encoders, target_encoder, used_features, best_params, acc = load_or_train_model(df)

    st.success("Fertig!")
    st.write("**Verwendete Merkmale:**", used_features)
    st.write("**Beste Parameter (XGB):**", best_params)
    st.write(f"**Testgenauigkeit (Accuracy):** {acc:.3f}")
    st.code(str(best_params), language="json")

    if MODEL_PATH.exists():
        st.info(f"Gespeichertes Modell: `{MODEL_PATH.name}` (Ordner: {MODEL_PATH.parent})")

with tab_predict:
    st.subheader("Einzelvorhersage")
    st.write("Gib Werte ein und erhalte eine *Risk*-Vorhersage.")
    # Lade Artefakte ohne Neu-Training
    try:
        model = joblib.load(MODEL_PATH)
        meta = joblib.load(MODEL_PATH.with_suffix('.meta.pkl'))
        encoders = meta['encoders']
        target_encoder = meta['target_encoder']
        features = meta['features']
    except Exception as e:
        st.error(f"Modell-Artefakte nicht gefunden. Bitte zuerst im Tab **Training** trainieren. ({e})")
        st.stop()

    cfg = get_feature_config(df, target_col="Risk")
    inputs = {}
    cols_left, cols_right = st.columns(2)

    for col, kind in cfg.items():
        if col not in features:  # nur tatsächlich im Modell verwendete Features anzeigen
            continue
        with (cols_left if list(cfg).index(col) % 2 == 0 else cols_right):
            if kind == "numeric":
                # sinnvolle Grenzen schätzen
                vmin = float(np.nanmin(df[col])) if col in df.columns else 0.0
                vmax = float(np.nanmax(df[col])) if col in df.columns else 100.0
                default = float(np.nanmedian(df[col])) if col in df.columns else 0.0
                inputs[col] = st.number_input(col, value=default, min_value=vmin, max_value=vmax, step=1.0, format="%.1f")
            else:
                # Kategorische Auswahl aus Trainingsdaten
                cats = encoders[col].classes_.tolist() if col in encoders else sorted(df[col].dropna().unique().tolist())
                default = cats[0] if cats else ""
                inputs[col] = st.selectbox(col, cats, index=0 if default in cats else 0)

    if st.button("Vorhersagen"):
        try:
            X_row = encode_features_for_inference(pd.DataFrame([inputs]), encoders, features)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_row)[0]
            pred = model.predict(X_row)[0]
            # zurück in Original-Labels der Zielvariable
            if target_encoder is not None:
                try:
                    inv = target_encoder.inverse_transform([pred])[0]
                except Exception:
                    inv = str(pred)
            else:
                inv = str(pred)

            st.success(f"**Vorhersage (Risk):** {inv}")
            if proba is not None and len(proba) == len(target_encoder.classes_):
                out = {cls: float(p) for cls, p in zip(target_encoder.classes_, proba)}
                st.write("Wahrscheinlichkeiten:", out)
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {e}")