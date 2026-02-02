import streamlit as st

st.set_page_config(page_title="Housing Market Dashboard", layout="centered")

st.title("🏠 Housing Market Risk Dashboard")

st.markdown("""
This dashboard shows a **weekly housing market signal**
designed for real estate investors and professionals.
""")

st.subheader("📍 Current Market Environment")
st.markdown("🔴 **High Risk / Caution**")

st.markdown("""
### How to read this:
- 🟢 Green = Supportive market  
- 🟡 Yellow = Mixed signals  
- 🔴 Red = Elevated risk  

This signal updates weekly.
""")
