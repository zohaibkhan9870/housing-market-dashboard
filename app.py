# ============================================================
# TAMPA HOUSING MARKET RISK DASHBOARD (STREAMLIT APP)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from pandas_datareader import data as pdr

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Tampa Housing Market Risk Dashboard",
    layout="centered"
)

st.title("🏠 Tampa Housing Market Risk Dashboard")
st.markdown(
    "This dashboard shows a **weekly housing market signal for Tampa, Florida**, "
    "designed for real estate investors and housing professionals."
)

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------
TARGET_METRO = "Tampa"

# ------------------------------------------------------------
# LOAD ZILLOW DATA (FILES MUST EXIST IN GITHUB REPO)
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
# SELECT TAMPA METRO
# ------------------------------------------------------------
price_matches = zillow_price[
    zillow_price["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
]
value_matches = zillow_value[
    zillow_value["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
]

if price_matches.empty or value_matches.empty:
    st.error("Tampa metro not found in Zillow files.")
    st.stop()

metro_name = price_matches["RegionName"].values[0]

price = pd.DataFrame(price_matches.iloc[0, 5:])
value = pd.DataFrame(value_matches.iloc[0, 5:])

# ------------------------------------------------------------
# LOAD FRED DATA (AUTO)
# ------------------------------------------------------------
@st.cache_data
def load_fred():
    start = "1950-01-01"
    end = datetime.today()

    fed = pd.concat([
        pdr.DataReader("MORTGAGE30US", "fred", start, end),
        pdr.DataReader("RRVRUSQ156N", "fred", start, end),
        pdr.DataReader("CPIAUCSL", "fred", start, end),
    ], axis=1)

    fed.columns = ["interest", "vacancy", "cpi"]
    fed = fed.sort_index().ffill().dropna()
    fed.index = fed.index + timedelta(days=2)
    return fed

fed_data = load_fred()

# ------------------------------------------------------------
# PREPARE PRICE DATA
# ------------------------------------------------------------
price.index = pd.to_datetime(price.index)
value.index = pd.to_datetime(value.index)

price["month"] = price.index.to_period("M")
value["month"] = value.index.to_period("M")

price_data = price.merge(value, on="month")
price_data.index = price.index
price_data.drop(columns=["month"], inplace=True)
price_data.columns = ["price", "value"]

price_data = fed_data.merge(
    price_data,
    left_index=True,
    right_index=True
)

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100

price_data["next_quarter"] = price_data["adj_price"].shift(-13)
price_data["change"] = (
    price_data["next_quarter"] > price_data["adj_price"]
).astype(int)

price_data["price_13w_change"] = price_data["adj_price"].pct_change(13)
price_data["value_52w_change"] = price_data["adj_value"].pct_change(52)

price_data.dropna(inplace=True)

# ------------------------------------------------------------
# WALK-FORWARD MODEL (SAFE FOR METROS)
# ------------------------------------------------------------
predictors = [
    "adj_price",
    "adj_value",
    "interest",
    "price_13w_change",
    "value_52w_change"
]

target = "change"

STEP = 52
START = max(104, int(len(price_data) * 0.5))

def predict_proba(train, test):
    rf = RandomForestClassifier(
        min_samples_split=10,
        random_state=1
    )
    rf.fit(train[predictors], train[target])
    return rf.predict_proba(test[predictors])[:, 1]

all_probs = []

for i in range(START, price_data.shape[0], STEP):
    train = price_data.iloc[:i]
    test = price_data.iloc[i:i+STEP]
    if len(test) > 0:
        all_probs.append(predict_proba(train, test))

if len(all_probs) == 0:
    st.error("Not enough historical data to generate signals for Tampa.")
    st.stop()

probs = np.concatenate(all_probs)

prob_data = price_data.iloc[START:].copy()
prob_data["prob_up"] = probs

# ------------------------------------------------------------
# REGIME LABELS
# ------------------------------------------------------------
def label_regime(p):
    if p > 0.65:
        return "Supportive Market"
    elif p < 0.45:
        return "High Risk / Caution"
    else:
        return "Mixed Signals"

prob_data["regime"] = prob_data["prob_up"].apply(label_regime)

latest = prob_data.tail(1)
prev = prob_data.tail(2).head(1)

current_regime = latest["regime"].values[0]
current_prob = latest["prob_up"].values[0]

# ------------------------------------------------------------
# DASHBOARD — CURRENT SIGNAL
# ------------------------------------------------------------
st.subheader("📍 Current Market Environment")
st.caption(f"Metro: {metro_name} | Updated: {latest.index[0].date()}")

if current_regime == "Supportive Market":
    st.success("🟢 Supportive Market\n\nConditions favor housing price stability or upside.")
elif current_regime == "Mixed Signals":
    st.warning("🟡 Mixed Signals\n\nMarket direction is unclear.")
else:
    st.error("🔴 High Risk / Caution\n\nDownside risk is elevated.")

# ------------------------------------------------------------
# WEEKLY CHANGE INDICATOR
# ------------------------------------------------------------
delta = current_prob - prev["prob_up"].values[0]

st.metric(
    "Weekly Outlook Change",
    f"{current_prob:.1%}",
    f"{delta:+.1%} vs last week"
)

# ------------------------------------------------------------
# PRICE + RISK CHART
# ------------------------------------------------------------
st.subheader("📈 Tampa Housing Price & Risk Signals")

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(
    prob_data.index,
    prob_data["adj_price"],
    color="black",
    linewidth=2,
    label="Real Home Price (Inflation-Adjusted)"
)

for i in range(len(prob_data) - 1):
    r = prob_data["regime"].iloc[i]
    if r == "Supportive Market":
        c = "green"
    elif r == "Mixed Signals":
        c = "gold"
    else:
        c = "red"

    ax.axvspan(
        prob_data.index[i],
        prob_data.index[i+1],
        color=c,
        alpha=0.12
    )

ax.set_title("Tampa Housing Market: Price Trend & Risk Signals")
ax.set_ylabel("Typical Home Price (Inflation-Adjusted)")
ax.set_xlabel("Date")

legend_items = [
    Patch(facecolor="green", alpha=0.25, label="Supportive Market"),
    Patch(facecolor="gold", alpha=0.25, label="Mixed Signals"),
    Patch(facecolor="red", alpha=0.25, label="High Risk / Caution")
]

ax.legend(
    handles=[plt.Line2D([0], [0], color="black", lw=2,
                        label="Real Home Price")] + legend_items,
    loc="upper left"
)

st.pyplot(fig)

# ------------------------------------------------------------
# HOW TO READ
# ------------------------------------------------------------
st.markdown("""
### How to read this chart
- **Black line:** Typical Tampa home prices adjusted for inflation  
- **Green background:** Supportive market conditions  
- **Yellow background:** Mixed or unclear signals  
- **Red background:** Elevated downside risk  

👉 Focus on **recent changes**, not old history.  
This signal updates **weekly**.
""")
