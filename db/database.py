# db/database.py

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import streamlit as st
from pathlib import Path

# point at a local file instead of Postgres
DB_FILE = Path("budget.db")
URL     = f"sqlite:///{DB_FILE}"

engine   = create_engine(URL, connect_args={"check_same_thread": False})
metadata = MetaData()

budget_table = Table(
    "budget_data", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("Plant", String, index=True),
    Column("Equipment", String),
    Column("Year", Integer, index=True),
    Column("Budget", Float),
    Column("MaintenanceType", String),
)

def init_db():
    metadata.create_all(engine)

def save_data(df: pd.DataFrame):
    try:
        df.to_sql("budget_data", engine, if_exists="append", index=False)
    except SQLAlchemyError as e:
        st.error(f"Error saving to DB: {e}")

def load_data() -> pd.DataFrame:
    try:
        return pd.read_sql_table("budget_data", engine)
    except SQLAlchemyError as e:
        st.error(f"Error loading from DB: {e}")
        return pd.DataFrame()
