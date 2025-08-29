import streamlit as st
import os
import joblib
import pandas as pd
import plotly.express as px
import numpy as np

@st.cache_resource
def load_artifacts():
    # Modell laden:
    model_files = ["xgb_credit_model.pkl", "extre_trees_credit_model.pkl"]
    model = None
    used_model_path = None
    for mf in model_files:
        if os.path.exists(mf):
            model = joblib.load(mf)
            used_model_path = mf
        break
    if model is None:
        st.error("Kein Modell gefunden, lege z.B. eine 'xgb_credit_model.pkl' Datei an.")
        st.stop()
    
    # Feature Reihenfolge bestimmen:
    feature_names_path = "feature_names.pkl"
    if os.path.exists(feature_names_path):
        feature_order = joblib.load(feature_names_path)
    else:
        st.warning("Keine Feature-Reihenfolge Datei gefunden!")
        st.stop()
    
    # Encoder laden:
    cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    encoders = {}
    missing = []
    for col in cat_cols:
        pkl = f"{col}_encoder.pkl"
        if os.path.exists(pkl):
            encoders[col] = joblib.load(pkl)
        else:
            missing.append(col)
    if missing:
        st.warning(f"Fehlende Encoder: {missing} - App funktioniert nicht!")
        st.stop()
    
    return model, used_model_path, encoders, feature_order

def options(col):
    enc = ENCODERS.get(col)
    return list(getattr(enc, "classes_")) if enc is not None else []

def encode_value(col, value):
    enc = ENCODERS.get(col)
    if enc is None:
        return value
    return enc.transform([value])[0]

# ------------------------------ #
# MAIN
# ------------------------------ #
# Grundlägende Konfiguration:
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳" ,layout="wide")
st.title("Credit Risk Prediction")
st.caption("Demo für den German Credit Data Datensatz")

# Modell & Encoders laden:
with st.spinner("Lade ML-Model..."):
    model, user_model_path, ENCODERS, FEATURE_ORDER =  load_artifacts() # (x, y, z), Tupel entpacken
if "model_toast_done" not in st.session_state:
    st.toast("Modell erfolgreich geladen!")
    st.session_state["model_toast_done"] = True

# Sidebar Menü:
with st.sidebar:
    st.header("Eingaben")
    
    # Einfache Defaults:
    age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
    sex = st.selectbox("Sex", options=options("Sex") or ["male", "female"])
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1, step=1)
    housing = st.selectbox("Housing", options=options("Housing"))
    saving_accounts = st.selectbox("Saving accounts", options=options("Saving accounts"))
    checking_account = st.selectbox("Checking account", options=options("Checking account"))
    purpose = st.selectbox("Purpose", options=options("Purpose"))
    credit_amount = st.number_input("Credit amount", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (months)", min_value=1, value=12, step=1)
    
    # Entscheidungsschwellee:
    st.divider()
    st.subheader("Einstellungen")
    threshold = st.slider("Entscheidungsschwelle ('Bad' ab):", 0.05, 0.95, 0.50, 0.01)
    

# Eine modellkonforme Eingabe erzeugen:
row = {
    "Age": age,
    "Sex": encode_value("Sex", sex),
    "Job": job,
    "Housing": encode_value("Housing", housing),
    "Saving accounts": encode_value("Saving accounts", saving_accounts),
    "Checking account": encode_value("Checking account", checking_account),
    "Credit amount": credit_amount,
    "Duration": duration,
    "Purpose": encode_value("Purpose", purpose)
}
input_df = pd.DataFrame([row])
input_df = input_df[FEATURE_ORDER]

# Navigationsmenü erstellen
tab_pred, tab_whatif, tab_explain, tab_about = st.tabs(["Predictions", "What-if", "Erklärungen", "About"])

# Vorhersagen (Predictions-Menü):
with tab_pred:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Modell-konformes Input")
        st.dataframe(input_df)
        st.info(f"Aktives Modell: **{user_model_path}**")
    with c2:
        st.subheader("Vorhersage")
        if st.button("Predict risk"):
            X = input_df
            proba = model.predict_proba(X)[0]
            p_bad = float(proba[1])
            is_bad = p_bad >= threshold
            st.metric("Bad-Risiko", f"{p_bad:.1%}")
            st.progress(min(max(p_bad, 0.0), 1.0))
            if is_bad:
                st.error(f"Einstuffung: **BAD** $\geq$ {threshold:.0%}")
            else:
                st.success(f"Einstuffung: **GOOD** $<$ {threshold:.0%}")
            
            with st.expander("Was bedeutet das?"):
                st.write(
                    """
                    * Das Modell gibt eine Wahrscheinlichkeit für 'bad' aus (Zahlungsauswall-Risiko).
                    * Über den Schwellenwert bestimmst du, ab wann aus der Wahrscheinlichkeit
                    eine binäre Entscheidung (GOOD/BAD) wird
                    """
                )
           
# Einfluss einzelner FEatures:      
with tab_whatif:
    st.subheader("What-if-Analyse: Wie ändert sich das Risiko, wenn ich ein Feature variiere?")
    feature_to_vary = st.selectbox("Feature wählen", FEATURE_ORDER)
    is_categorical = feature_to_vary in ENCODERS
    base = input_df.iloc[0].copy()

    if is_categorical:
        cats = list(ENCODERS[feature_to_vary].classes_)
        picked_cat = st.selectbox("Wert setzen", cats)
        vary_values_enc = ENCODERS[feature_to_vary].transform(cats)
    
    else:
    # aktuellen Wert des Features holen
        current_val = float(base[feature_to_vary])
        c1, c2 = st.columns(2)
        with c1:
            # Standardwerte für Min/Max-Bereich
            min_default = max(0.0, current_val * 0.5)
            max_default = max(current_val * 1.5, current_val + 1.0)

            # Debug-Ausgaben (kannst du entfernen)
            print(min_default)
            print(max_default)

            # Slider für Wertebereich
            min_val, max_val = st.slider(
                "Bereich",
                min_value=0.0,
                max_value=float(max_default * 2),
                value=(float(min_default), float(max_default)),
                step=1.0
            )
        with c2:
            # Slider für Anzahl der Schritte
            steps = st.slider("Schritte", 3, 50, 11)

        # gleichmäßig verteilte Werte zwischen Min und Max erzeugen
        vary_values_raw = np.linspace(min_val, max_val, int(steps))

        # Labels (für Anzeige auf Achse)
        cats = [float(v) for v in vary_values_raw]

        # Für Modell: dieselben Werte (numerisch)
        vary_values_enc = vary_values_raw
    
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("What-if berechnen"):
            st.session_state["whatif_active"] = True
    with b2:
        if st.button("Auto-Update stoppen"):
            st.session_state["whatif_active"] = False
        
    if st.session_state["whatif_active"]:
        if (min_val is not None) and (max_val is not None) and (max_val <= min_val):
            st.warning("Max muss größer als Min sein.")
        else:
            probs = []
            labels = []
            for val_raw, val_enc in zip(cats, vary_values_enc):
                row_mut = base.copy()
                row_mut[feature_to_vary] = val_enc
                X_mut = pd.DataFrame([row_mut])[FEATURE_ORDER]
                p_bad = model.predict_proba(X_mut)[0][1]
                probs.append(p_bad)
                labels.append(val_raw)
                
            chart_df = pd.DataFrame({
                "Wert": labels,
                "Bad-Risiko": probs
            })
            chart_df.set_index("Wert")

            fig = px.bar(
                chart_df,
                x="Wert",
                y="Bad-Risiko",
                text="Bad-Risiko",
                color="Bad-Risiko"
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(
                yaxis=dict(title="Risiko"),
                xaxis=dict(title=feature_to_vary),
                title=f"What-if Analyse für {feature_to_vary}"
            )
            st.plotly_chart(fig)
    
    