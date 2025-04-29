# app.py

import os
import json
import time
import joblib
import torch

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from streamlit_option_menu import option_menu

from models.lstm_model import BayesianMultiLSTM, train_bayesian_lstm, predict_with_uncertainty

# --- Credentials & Auth ---
creds = st.secrets["credentials"]
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.sidebar.subheader("🔒 Please log in")
    user = st.sidebar.text_input("Username")
    pwd  = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Log in"):
        if user in creds and creds[user] == pwd:
            st.session_state.authenticated = True
            st.session_state.user_name = user
            st.sidebar.success(f"Welcome, {user}!")
        else:
            st.sidebar.error("❌ Invalid credentials")
    st.stop()

# --- Config & Gemini ---
st.set_page_config(page_title="Budget Simulator", layout="wide")
gem_api   = st.secrets["gemini"]["api_key"]
genai.configure(api_key=gem_api)
assistant = genai.GenerativeModel("models/gemini-2.0-flash")

def ask_gemini(prompt: str, payload: dict) -> str:
    try:
        resp = assistant.generate_content([prompt, json.dumps(payload)])
        return resp.text
    except Exception as e:
        return f"Gemini error: {e}"

# --- Helper: Load or Train & Save Model ---
def get_or_train_model(plant: str, features_past: pd.DataFrame, budget_cols: list):
    os.makedirs("model_weights", exist_ok=True)
    wpath = f"model_weights/{plant}_model.pt"
    spath = f"model_weights/{plant}_scaler.pkl"

    model = BayesianMultiLSTM(
        input_size=features_past.shape[1],
        hidden_size=50,
        output_size=len(budget_cols),
        dropout=0.3
    )

    if os.path.exists(wpath) and os.path.exists(spath):
        scaler = joblib.load(spath)
        model.load_state_dict(torch.load(wpath, weights_only=True))
    else:
        model, scaler = train_bayesian_lstm(
            df_features=features_past,
            budget_cols=budget_cols,
            lookback=3, epochs=50, batch_size=16,
            hidden_size=50, dropout=0.3
        )
        torch.save(model.state_dict(), wpath)
        joblib.dump(scaler, spath)

    model.eval()
    return model, scaler


# --- Sidebar Menu ---
with st.sidebar:
    st.title("BUDGET SIMULATOR")
    choice = option_menu(
        menu_title=None,
        options=["Dashboard", "Input Data", "Help", "Settings"],
        icons=["house", "cloud-upload", "question-circle", "gear"],
        default_index=0,
    )
    st.markdown("---")

# --- Input Data ---
if choice == "Input Data":
    st.header("📥 Upload Budget Data")
    uploaded = st.file_uploader("Excel (long format)", type="xlsx")
    if uploaded:
        df = pd.read_excel(uploaded)
        required = {"Plant", "Equipment", "Year", "Budget", "MaintenanceType"}
        if not required.issubset(df.columns):
            st.error(f"Need columns: {required}")
        else:
            df["MaintenanceType"] = df["MaintenanceType"].fillna("Unknown")
            st.session_state["df"] = df.copy()
            st.success("Data loaded!")
    st.stop()

# ensure data exists
if "df" not in st.session_state:
    st.warning("Upload your data in Input Data first.")
    st.stop()
df = st.session_state["df"]

# --- Dashboard ---
if choice == "Dashboard":
    st.header(f"Hello {st.session_state.get('user_name','Admin')}")
    plant    = st.selectbox("Power Plant", df["Plant"].unique())
    plant_df = df[df["Plant"] == plant]

    # years
    pmn, pmx = int(plant_df.Year.min()), int(plant_df.Year.max())
    past_years   = list(range(pmn, pmx+1))
    future_years = list(range(pmx+1, 2041))

    # actual pivot
    actual_pivot = (
        plant_df
        .pivot_table(index="Year", columns="Equipment", values="Budget", aggfunc="sum")
        .reindex(past_years, fill_value=0)
    )

    # maintenance & cycle
    mt = (
        plant_df[["Year","MaintenanceType"]]
        .drop_duplicates("Year")
        .set_index("Year")
        .reindex(past_years)
        .fillna("Unknown")
    )
    maint_ohe = pd.get_dummies(mt["MaintenanceType"])
    cycle = ["2-year Minor","Forced","4-year Minor","Forced",
             "2-year Minor","Forced","8-year Major","Forced"]
    majors = mt[mt=="8-year Major"].dropna().index
    last_major = majors.max() if len(majors) else pmn
    idx0 = cycle.index("8-year Major")
    idxs_p = [(idx0 + (y-last_major))%len(cycle) for y in past_years]
    cycle_df_p = pd.DataFrame({"cycle_idx": idxs_p}, index=past_years)
    idxs_f = [(idx0 + (y-last_major))%len(cycle) for y in future_years]
    cycle_df_f = pd.DataFrame({"cycle_idx": idxs_f}, index=future_years)

    # features
    features_past = pd.concat([actual_pivot, maint_ohe, cycle_df_p], axis=1)
    budget_cols   = actual_pivot.columns.tolist()
    zero          = pd.DataFrame(0, index=future_years, columns=budget_cols)
    future_ohe    = pd.get_dummies(
        pd.Series(["Unknown"]*len(future_years), index=future_years),
        prefix="", prefix_sep=""
    ).reindex(columns=maint_ohe.columns, fill_value=0)
    features_fut  = pd.concat([zero, future_ohe, cycle_df_f], axis=1)

    # load/train once
    model, scaler = get_or_train_model(plant, features_past, budget_cols)

    # forecast
    mean_df, std_df = predict_with_uncertainty(
        features_past, features_fut,
        model, scaler,
        budget_cols, lookback=3, mc_samples=50
    )

    # metrics & chart
    st.metric("Expenses So Far",      f"${actual_pivot.sum(axis=1).iloc[-1]:,.0f}")
    st.metric("Next Year Budget",     f"${mean_df.sum(axis=1).iloc[0]:,.0f}")
    st.markdown("---")
    view = st.selectbox("Show:", ["Forecast only","Past only","Both"])
    fig = go.Figure()
    if view in ("Both","Past only"):
        for eq in budget_cols:
            fig.add_bar(x=actual_pivot.index, y=actual_pivot[eq],
                        name=f"Actual {eq}", marker_opacity=0.6)
    if view in ("Both","Forecast only"):
        for eq in budget_cols:
            fig.add_bar(x=mean_df.index, y=mean_df[eq], name=f"Forecast {eq}",
                        error_y=dict(type="data", array=std_df[eq]))
    fig.update_layout(barmode="group", xaxis_title="Year",
                      yaxis_title="Budget (USD)", legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    # Gemini Assistant
    st.markdown("---")
    st.subheader("Gemini Assistant")
    if st.checkbox("Enable Gemini Assistant"):
        payload = {
            "actual":     actual_pivot.reset_index().to_dict(orient="records"),
            "forecast":   mean_df.reset_index().to_dict(orient="records"),
            "uncertainty":std_df.reset_index().to_dict(orient="records")
        }
        question = st.text_input("Ask your budget question…")
        if st.button("Ask Gemini"):
            st.info("Sending to Gemini…")
            prompt = "Here is my budget data. Please answer: " + question
            answer = ask_gemini(prompt, payload)
            st.success("Gemini’s answer:")
            st.markdown(answer)

    # Work Orders
    st.markdown("---")
    with st.expander("➕ Add Work Order Entry", expanded=False):
        with st.form("add_entry"):
            c1, c2 = st.columns(2)
            with c1:
                np_plant = st.selectbox("Plant", df["Plant"].unique(), key="in_plant")
                np_equip = st.selectbox("Equipment", df["Equipment"].unique(), key="in_equip")
                np_year  = st.number_input("Year", min_value=pmn, max_value=2040,
                                           value=pmx+1, key="in_year")
            with c2:
                np_amt   = st.number_input("Budget", min_value=0.0, value=0.0, key="in_amt")
                np_mtype = st.selectbox("MaintenanceType",
                                        df["MaintenanceType"].unique(), key="in_mtype")
            if st.form_submit_button("Save Entry"):
                from db.database import save_data
                new = pd.DataFrame([{
                    "Plant": np_plant, "Equipment": np_equip,
                    "Year": np_year, "Budget": np_amt,
                    "MaintenanceType": np_mtype
                }])
                save_data(new)
                st.success("Saved—reloading…")
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("Work Order (Predicted)")
    entries = []
    for y in mean_df.index:
        for eq in budget_cols:
            idx = (idx0 + (y-last_major)) % len(cycle)
            entries.append({
                "Equipment":      eq,
                "Year":           y,
                "MaintenanceType": cycle[idx],
                "Budget":         mean_df.loc[y, eq],
                "Uncertainty":    std_df.loc[y, eq]
            })
    work_df = pd.DataFrame(entries).sort_values("Year", ascending=True)
    fmt = {"Budget":"${:,.0f}", "Uncertainty":"${:,.0f}"}
    st.dataframe(work_df.style.format(fmt), height=400, use_container_width=True)

elif choice == "Help":
    st.header("🆘 Help")
    st.write("Contact amirulhaq@ft.um-surabaya.ac.id")

elif choice == "Settings":
    st.header("⚙️ Settings")
    st.write("Nothing here yet.")
