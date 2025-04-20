# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import json
from streamlit_option_menu import option_menu

from models.lstm_model import train_bayesian_lstm, predict_with_uncertainty

# --- Config & Gemini ---
st.set_page_config(page_title="Budget Simulator", layout="wide")
gem_api   = st.secrets["gemini"]["api_key"]
genai.configure(api_key=gem_api)
gem_model = genai.GenerativeModel("models/text-bison-001")

# Replace your old setup…
# gem_model = genai.GenerativeModel("models/text-bison-001")

# …with chat completion:
def ask_gemini(prompt: str) -> str:
    try:
        resp = genai.chat.completions.create(
            model="models/chat-bison-001",
            messages=[{"author": "user", "content": prompt}],
        )
        return resp.candidates[0].message.content
    except Exception as e:
        for m in genai.list_models():
            # supported_methods might include "generate_content" or "chat"
            print(m.name, m.supported_methods)
        return f"Gemini error: {e}"


# --- Sidebar ---
with st.sidebar:
    st.title("BUDGET SIMULATOR")
    choice = option_menu(
        menu_title=None,
        options=["Dashboard", "Input Data", "Help", "Settings"],
        icons=["house", "cloud-upload", "question-circle", "gear"],
        default_index=0,
    )
    st.markdown("---")
    st.button("Log in")

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

if "df" not in st.session_state:
    st.warning("Upload your data in Input Data first.")
    st.stop()
df = st.session_state["df"]

# --- Dashboard ---
if choice == "Dashboard":
    st.header(f"Hello {st.session_state.get('user_name','Admin')}")
    plant    = st.selectbox("Power Plant", df["Plant"].unique())
    plant_df = df[df["Plant"] == plant]

    # Historical and future year ranges
    plant_min    = int(plant_df.Year.min())
    plant_max    = int(plant_df.Year.max())
    past_years   = list(range(plant_min, plant_max + 1))
    future_years = list(range(plant_max + 1, 2041))

    # Actual budgets pivot
    actual_pivot = (
        plant_df
        .pivot_table(index="Year", columns="Equipment", values="Budget", aggfunc="sum")
        .reindex(past_years, fill_value=0)
    )

    # One MaintenanceType per year
    mt = (
        plant_df[["Year","MaintenanceType"]]
        .drop_duplicates("Year")
        .set_index("Year")
        .reindex(past_years)
        .fillna("Unknown")
    )
    maint_ohe = pd.get_dummies(mt["MaintenanceType"])
    maint_cols = maint_ohe.columns.tolist()

    # Define 8-year cycle and anchor on last major outage
    cycle = [
        "2-year Minor","Forced","4-year Minor","Forced",
        "2-year Minor","Forced","8-year Major","Forced"
    ]
    majors = mt[mt["MaintenanceType"]=="8-year Major"].index
    last_major_year = majors.max() if len(majors) else plant_min
    major_idx = cycle.index("8-year Major")

    # Build cycle_idx for past and future
    cycle_idx_past = [(major_idx + (y - last_major_year))%len(cycle) for y in past_years]
    cycle_df_past  = pd.DataFrame({"cycle_idx": cycle_idx_past}, index=past_years)
    cycle_idx_fut  = [(major_idx + (y - last_major_year))%len(cycle) for y in future_years]
    cycle_df_fut   = pd.DataFrame({"cycle_idx": cycle_idx_fut}, index=future_years)

    # Assemble past features
    features_past = pd.concat([actual_pivot, maint_ohe, cycle_df_past], axis=1)

    # Train Bayesian LSTM
    budget_cols = actual_pivot.columns.tolist()
    model, scaler = train_bayesian_lstm(
        df_features=features_past,
        budget_cols=budget_cols,
        lookback=3, epochs=50, batch_size=16,
        hidden_size=50, dropout=0.3
    )

    # Build future placeholder features
    zero_budg   = pd.DataFrame(0, index=future_years, columns=budget_cols)
    future_mt   = pd.DataFrame({"MaintenanceType":["Unknown"]*len(future_years)}, index=future_years)
    future_ohe  = pd.get_dummies(future_mt["MaintenanceType"]).reindex(columns=maint_cols, fill_value=0)
    features_fut = pd.concat([zero_budg, future_ohe, cycle_df_fut], axis=1)

    # Forecast with uncertainty
    mean_df, std_df = predict_with_uncertainty(
        features_past, features_fut, model, scaler,
        budget_cols, 3, 50
    )

    # Metrics
    st.metric("Expenses So Far", f"${actual_pivot.sum(axis=1).iloc[-1]:,.0f}")
    st.metric("Next Year Budget", f"${mean_df.sum(axis=1).iloc[0]:,.0f}")

    # Chart with selector
    st.markdown("---")
    st.subheader("Budget 2013–2040")
    view = st.selectbox("Show:", ["Forecast only", "Past only", "Both"])
    fig = go.Figure()
    if view in ("Both","Past only"):
        for eq in budget_cols:
            fig.add_trace(go.Bar(
                x=actual_pivot.index,
                y=actual_pivot[eq],
                name=f"Actual {eq}", marker_opacity=0.6
            ))
    if view in ("Both","Forecast only"):
        for eq in budget_cols:
            fig.add_trace(go.Bar(
                x=mean_df.index,
                y=mean_df[eq],
                name=f"Forecast {eq}",
                error_y=dict(type="data", array=std_df[eq])
            ))
    fig.update_layout(barmode="group", xaxis_title="Year", yaxis_title="Budget (USD)", legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    # Gemini Assistant
    st.markdown("---")
    st.subheader("Gemini Assistant")

    # Enable toggle
    enable = st.checkbox("Enable Gemini Assistant", value=False)
    if enable:
        # Show what we’ll send
        st.markdown("**Payload to Gemini:**")
        payload = {
            "actual":    actual_pivot.reset_index().to_dict(orient="records"),
            "forecast":  mean_df.reset_index().to_dict(orient="records"),
            "uncertainty": std_df.reset_index().to_dict(orient="records")
        }
        st.write(payload)  # optional visibility

        # User question input
        question = st.text_input("Ask your budget question…")
        if st.button("Ask Gemini"):
            st.info("Sending data to Gemini…")
            # Reuse your configured model
            assistant = genai.GenerativeModel("models/gemini-2.0-flash")
            # First element: instruction; second: the JSON payload
            response = assistant.generate_content([
                "Here is my actual and forecast budget data. " 
                + "Please summarize key trends or anomalies.",
                json.dumps(payload)
            ])
            st.success("Gemini’s answer:")
            st.markdown(response.text)
    else:
        st.write("Toggle **Enable Gemini Assistant** to ask questions.")


    # Add Work Order Entry
    st.markdown("---")
    with st.expander("➕ Add Work Order Entry", expanded=False):
        with st.form("add_entry"):
            c1, c2 = st.columns(2)
            with c1:
                np_plant = st.selectbox("Plant", df["Plant"].unique(), key="in_plant")
                np_equip = st.selectbox("Equipment", df["Equipment"].unique(), key="in_equip")
                np_year  = st.number_input("Year", min_value=plant_min, max_value=2040, value=plant_max+1, key="in_year")
            with c2:
                np_amt   = st.number_input("Budget", min_value=0.0, value=0.0, key="in_amt")
                np_mtype = st.selectbox("MaintenanceType", df["MaintenanceType"].unique(), key="in_mtype")
            if st.form_submit_button("Save Entry"):
                from db.database import save_data
                new = pd.DataFrame([{
                    "Plant": np_plant,
                    "Equipment": np_equip,
                    "Year": np_year,
                    "Budget": np_amt,
                    "MaintenanceType": np_mtype
                }])
                save_data(new)
                st.success("Saved—reloading…")
                st.experimental_rerun()

    # Work Order Table with Uncertainty
    st.markdown("---")
    st.subheader("Work Order (Predicted)")
    entries = []
    for y in mean_df.index:
        for eq in budget_cols:
            mtype = cycle[(major_idx + (y - last_major_year)) % len(cycle)]
            entries.append({
                "Equipment": eq,
                "Year": y,
                "MaintenanceType": mtype,
                "Budget": mean_df.loc[y, eq],
                "Uncertainty": std_df.loc[y, eq]
            })
    work_df = pd.DataFrame(entries).sort_values("Year", ascending=True)
    # format USD columns
    fmt = {"Budget":"${:,.0f}", "Uncertainty":"${:,.0f}"}
    st.dataframe(work_df.style.format(fmt), height=400, use_container_width=True)

elif choice == "Help":
    st.header("🆘 Help")
    st.write("Contact amirulhaq@ft.um-surabaya.ac.id")
elif choice == "Settings":
    st.header("⚙️ Settings")
    st.write("Nothing here yet.")
