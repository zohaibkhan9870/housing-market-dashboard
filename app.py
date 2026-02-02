import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Tampa Housing Market Risk",
    layout="centered"
)

st.title("🏠 Tampa Housing Market Risk Dashboard")
st.markdown(
    "This dashboard shows a **weekly housing market risk signal for Tampa, Florida**, "
    "designed for real estate investors and housing professionals."
)

# -----------------------------
# LOAD FRED DATA (AUTO-UPDATES)
# -----------------------------
start = "1950-01-01"
end = datetime.today()

fed_data = pd.concat([
    pdr.DataReader("MORTGAGE30US", "fred", start, end),
    pdr.DataReader("RRVRUSQ156N", "fred", start, end),
    pdr.DataReader("CPIAUCSL", "fred", start, end),
], axis=1)

fed_data.columns = ["interest", "vacancy", "cpi"]
fed_data = fed_data.sort_index().ffill().dropna()
fed_data.index = fed_data.index + timedelta(days=2)

# -----------------------------
# LOAD ZILLOW DATA (TAMPA)
# -----------------------------
zillow_price = pd.read_csv("Metro_median_sale_price_uc_sfrcondo_sm_week.csv")
zillow_value = pd.read_csv("Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")

price_row = zillow_price[zillow_price["RegionName"] == "Tampa, FL"]
value_row = zillow_value[zillow_value["RegionName"] == "Tampa, FL"]

price = pd.DataFrame(price_row.iloc[0, 5:])
value = pd.DataFrame(value_row.iloc[0, 5:])

price.index = pd.to_datetime(price.index)
value.index = pd.to_datetime(value.index)

price["month"] = price.index.to_period("M")
value["month"] = value.index.to_period("M")

price_data = price.merge(value, on="month")
price_data.index = price.index
price_data.columns = ["price", "value"]

# -----------------------------
# MERGE WITH FRED
# -----------------------------
price_data = fed_data.merge(price_data, left_index=True, right_index=True)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100

price_data["next_quarter"] = price_data["adj_price"].shift(-13)
price_data["change"] = (price_data["next_quarter"] > price_data["adj_price"]).astype(int)

price_data["price_13w_change"] = price_data["adj_price"].pct_change(13)
price_data["value_52w_change"] = price_data["adj_value"].pct_change(52)

price_data.dropna(inplace=True)

# -----------------------------
# MODEL
# -----------------------------
predictors = [
    "adj_price",
    "adj_value",
    "interest",
    "price_13w_change",
    "value_52w_change"
]

target = "change"

rf = RandomForestClassifier(
    min_samples_split=10,
    random_state=1
)

rf.fit(price_data[predictors], price_data[target])
price_data["prob_up"] = rf.predict_proba(price_data[predictors])[:, 1]

def label_regime(p):
    if p > 0.65:
        return "Supportive Market"
    elif p < 0.45:
        return "High Risk / Caution"
    else:
        return "Mixed Signals"

price_data["regime"] = price_data["prob_up"].apply(label_regime)

# -----------------------------
# CURRENT SIGNAL
# -----------------------------
latest = price_data.iloc[-1]
previous = price_data.iloc[-2]

st.subheader("📍 Current Market Environment")
st.caption(f"Metro: Tampa, FL | Updated: {latest.name.date()}")

st.markdown(f"### 🔴 {latest['regime']}")

# -----------------------------
# WEEKLY CHANGE INDICATOR
# -----------------------------
delta = latest["prob_up"] - previous["prob_up"]

if delta > 0.03:
    change_text = "⬆️ Improving vs last week"
elif delta < -0.03:
    change_text = "⬇️ Deteriorating vs last week"
else:
    change_text = "➡️ No meaningful change vs last week"

st.info(change_text)

# -----------------------------
# CHART (LAST 5 YEARS)
# -----------------------------
st.subheader("📈 Home Prices & Risk Signals (Last 5 Years)")

plot_data = price_data.last("5Y")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    plot_data.index,
    plot_data["adj_price"],
    color="black",
    linewidth=2,
    label="Real Home Price (Inflation-Adjusted)"
)

for i in range(len(plot_data) - 1):
    regime = plot_data["regime"].iloc[i]
    color = (
        "green" if regime == "Supportive Market"
        else "yellow" if regime == "Mixed Signals"
        else "red"
    )
    ax.axvspan(
        plot_data.index[i],
        plot_data.index[i+1],
        color=color,
        alpha=0.15
    )

ax.set_title("Tampa Housing Market: Price Trend & Risk Signals")
ax.set_ylabel("Real Home Price")
ax.legend()

plt.tight_layout()
st.pyplot(fig)

# -----------------------------
# HOW TO READ
# -----------------------------
st.markdown("""
### How to read this chart:
- **Black line**: Typical Tampa home price (inflation-adjusted)
- **Green background**: Supportive market conditions
- **Yellow background**: Mi
""")
