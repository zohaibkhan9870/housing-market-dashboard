import streamlit as st
import pandas as pd
from datetime import date

# ------------------------------
# PAGE SETUP
# ------------------------------
st.set_page_config(
    page_title="Tampa Housing Market Signal",
    layout="centered"
)

# ------------------------------
# HEADER
# ------------------------------
st.title("🏠 Tampa Housing Market Risk Dashboard")

st.markdown("""
This dashboard shows a **weekly housing market signal**
for **Tampa, Florida**, designed for real estate investors
and housing professionals.
""")

# ------------------------------
# MODEL METADATA (PLACEHOLDER FOR NOW)
# ------------------------------
LAST_UPDATE = date.today()
METRO = "Tampa, FL"

# These will later come from your real model
prob_up = 0.32
regime = "Risk"

# ------------------------------
# MAIN SIGNAL DISPLAY
# ------------------------------
st.subheader("📍 Current Market Environment")
st.caption(f"Metro: **{METRO}** | Updated: **{LAST_UPDATE}**")

if regime == "Bull":
    st.success("🟢 Supportive Market")
    explanation = (
        "Conditions appear supportive for housing prices "
        "over the next quarter."
    )
elif regime == "Neutral":
    st.warning("🟡 Mixed Signals")
    explanation = (
        "The housing outlook is mixed, with no strong directional bias."
    )
else:
    st.error("🔴 High Risk / Caution")
    explanation = (
        "Downside risk is elevated. Caution is warranted "
        "for near-term housing prices."
    )

st.markdown(explanation)

# ------------------------------
# HOW TO READ SECTION
# ------------------------------
st.markdown("---")
st.subheader("ℹ️ How to Read This Signal")

st.markdown("""
- 🟢 **Green** = Supportive housing environment  
- 🟡 **Yellow** = Mixed or uncertain conditions  
- 🔴 **Red** = Elevated downside risk  

This signal is designed to be:
- Forward-looking (next ~3 months)
- Updated weekly
- Easy to interpret without technical knowledge
""")

# ------------------------------
# FOOTER
# ------------------------------
st.caption(
    "This tool is for informational purposes only and does not "
    "constitute investment advice."
)
