"""Private Client Portfolio Optimizer — Streamlit UI only (no heavy logic here)."""
import streamlit as st

st.set_page_config(page_title="Private Client Portfolio Optimizer", layout="wide")
st.title("Private Client Portfolio Optimizer")
st.info("The app is being rebuilt for the new methodology (maximum return under a volatility cap, "
        "USD multi-asset universe). It will be back with five tabs: Client Profile, Constraints, Portfolio, "
        "Backtest and Simulation.")
