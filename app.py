# ============================================================
# TAMPA HOUSING MARKET RISK DASHBOARD (STREAMLIT CLOUD READY)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Tampa Housing Market Risk Dashboard",
    layout="centered"
)

st.title("🏠 Tampa Housing Market Risk Dashboard")
st.write(
    "This dashboard shows a **weekly housing market signal for Tampa, Florida**, "
    "designed for real estate investors and housing professionals."
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
TARGET_METRO = "Tampa"
START = 260
STEP = 52

# ------------------------------------------------------------
# LOAD ZILLOW DATA (FILES MUST EXIST IN REPO)
# ------------------------------------------------------------
@st.cache_data
def load_zillow():
    price = pd.read_csv(
        "Metro_median_sale_price_uc_sfrcondo_sm_week.csv"
    )
    value = pd.read_csv(
        "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    )
    return price, value

zillow_price, zillow_value = load_zillow()

# ------------------------------------------------------------
# SELECT METRO
# ------------------------------------------------------------
price_row = zillow_price[
    zillow_price["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
]

value_row = zillow_value[
    zillow_value["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
]

if price_row.empty or value_row.empty:
    st.error("Tampa metro not found in Zillow files.")
    st.stop()

price = pd.DataFrame(price_row.iloc[0, 5:])
value = pd.DataFrame(value_row.iloc[0, 5:])

price.index = pd.to_datetime(price.index)
value.index = pd.to_datetime(value.index)

price["month"] = price.index.to_period("M")
value["month"] = value.index.to_period("M")

price_data = price.merge(value, on="month")
price_data.index = price.index
price_data.columns = ["price", "value"]

# ------------------------------------------------------------
# LOAD FRED DATA (CSV METHOD — SAFE)
# ------------------------------------------------------------
@st.cache_data
def load_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

mortgage = load_fred_series("MORTGAGE30US")
vacancy = load_fred_series("RRVRUSQ156N")
cpi = load_fred_series("CPIAUCSL")

fed_data = mortgage.join(vacancy).join(cpi)
fed_data.columns = ["interest", "vacancy", "cpi"]
fed_data = fed_data.ffill().dropna()

# ------------------------------------------------------------
# MERGE DATA
# ------------------------------------------------------------
data = fed_data.merge(price_data, left_index=True, right_index=True)

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
data["adj_price"] = data["price"] / data["cpi"] * 100
data["adj_value"] = data["value"] / data["cpi"] * 100

data["next_quarter"] = data["adj_price"].shift(-13)
data["change"] = (data["next_quarter"] > data["adj_price"]).astype(int)

data["price_13w_change"] = data["adj_price"].pct_change(13)
data["value_52w_change"] = data["adj_value"].pct_change(52)

data.dropna(inplace=True)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
predictors = [
    "adj_price",
    "adj_value",
    "interest",
    "price_13w_change",
    "value_52w_change",
]

all_probs = []

for i in range(START, len(data), STEP):
    train = data.iloc[:i]
    test = data.iloc[i:i + STEP]

    if len(test) == 0:
        continue

    rf = RandomForestClassifier(
        min_samples_split=10,
        random_state=1
    )
    rf.fit(train[predictors], train["change"])
    all_probs.append(rf.predict_proba(test[predictors])[:, 1])

if len(all_probs) == 0:
    st.error("Not enough data to run model.")
    st.stop()

probs = np.concatenate(all_probs)

prob_data = data.iloc[START:].copy()
prob_data["prob_up"] = probs

def label_regime(p):
    if p > 0.65:
        return "Supportive"
    elif p < 0.45:
        return "High Risk"
    else:
        return "Mixed"

prob_data["regime"] = prob_data["prob_up"].apply(label_regime)

# ------------------------------------------------------------
# CURRENT SIGNAL
# ------------------------------------------------------------
latest = prob_data.iloc[-1]
prev = prob_data.iloc[-2]

st.subheader("📍 Current Market Environment")

if latest["regime"] == "Supportive":
    st.success("🟢 Supportive Market")
elif latest["regime"] == "Mixed":
    st.warning("🟡 Mixed Signals")
else:
    st.error("🔴 High Risk / Caution")

delta = latest["prob_up"] - prev["prob_up"]

st.metric(
    label="This Week vs Last Week",
    value=f"{latest['prob_up']:.2f}",
    delta=f"{delta:+.2f}"
)

# ------------------------------------------------------------
# PRICE + RISK CHART
# ------------------------------------------------------------
st.subheader("📈 Tampa Home Prices & Risk Signals")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    prob_data.index,
    prob_data["adj_price"],
    color="black",
    linewidth=2,
    label="Real Home Price (Inflation-Adjusted)"
)

for i in range(len(prob_data) - 1):
    color = (
        "green" if prob_data["regime"].iloc[i] == "Supportive"
        else "gold" if prob_data["regime"].iloc[i] == "Mixed"
        else "red"
    )
    ax.axvspan(
        prob_data.index[i],
        prob_data.index[i + 1],
        color=color,
        alpha=0.12
    )

ax.set_title("Tampa Housing Market: Price Trend & Risk Signals")
ax.set_ylabel("Inflation-Adjusted Home Price")
ax.legend()

st.pyplot(fig)

# --------------
