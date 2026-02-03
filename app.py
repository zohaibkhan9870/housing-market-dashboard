# ============================================================
# TAMPA HOUSING MARKET RISK DASHBOARD (STREAMLIT CLOUD SAFE)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
# LOAD FRED DATA (NO pandas_datareader)
# ------------------------------------------------------------
@st.cache_data
def load_fred():
    def fred(series):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        df = pd.read_csv(url)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df.set_index("DATE", inplace=True)
        df.replace(".", np.nan, inplace=True)
        return df.astype(float)

    mortgage = fred("MORTGAGE30US")
    vacancy = fred("RRVRUSQ156N")
    cpi = fred("CPIAUCSL")

    fed = mortgage.join(vacancy, how="outer").join(cpi, how="outer")
    fed.columns = ["interest", "vacancy", "cpi"]
    fed = fed.ffill().dropna()

    # Align weekly timing
    fed.index = fed.index + pd.Timedelta(days=2)
    return fed


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


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
fed_data = load_fred()
zillow_price, zillow_value = load_zillow()

TARGET_METRO = "Tampa"

price_matches = zillow_price[
    zillow_price["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
]
value_matches = zillow_value[
    zillow_value["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
]

if price_matches.empty or value_matches.empty:
    st.error("Tampa metro not found in Zillow files.")
    st.stop()

price = pd.DataFrame(price_matches.iloc[0, 5:])
value = pd.DataFrame(value_matches.iloc[0, 5:])

# ------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------
price.index = pd.to_datetime(price.index)
value.index = pd.to_datetime(value.index)

price["month"] = price.index.to_period("M")
value["month"] = value.index.to_period("M")

price_data = price.merge(value, on="month")
price_data.index = price.index
price_data.columns = ["price", "value"]
price_data.drop(columns=["month"], inplace=True)

price_data = fed_data.merge(price_data, left_index=True, right_index=True)

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
# WALK-FORWARD MODEL
# ------------------------------------------------------------
predictors = [
    "adj_price",
    "adj_value",
    "interest",
    "price_13w_change",
    "value_52w_change",
]

START = 260
STEP = 52

def predict_proba(train, test):
    rf = RandomForestClassifier(
        min_samples_split=10,
        random_state=1
    )
    rf.fit(train[predictors], train["change"])
    return rf.predict_proba(test[predictors])[:, 1]

all_probs = []

for i in range(START, len(price_data), STEP):
    train = price_data.iloc[:i]
    test = price_data.iloc[i:i + STEP]
    if len(test) > 0:
        all_probs.append(predict_proba(train, test))

probs = np.concatenate(all_probs)

prob_data = price_data.iloc[START:].copy()
prob_data["prob_up"] = probs

# ------------------------------------------------------------
# REGIME LABEL
# ------------------------------------------------------------
def label_regime(p):
    if p > 0.65:
        return "Supportive Market"
    elif p < 0.45:
        return "High Risk / Caution"
    else:
        return "Mixed Signals"

prob_data["regime"] = prob_data["prob_up"].apply(label_regime)

# ------------------------------------------------------------
# CURRENT SIGNAL
# ------------------------------------------------------------
latest = prob_data.iloc[-1]
prev = prob_data.iloc[-2]

st.subheader("📍 Current Market Environment")

st.markdown(f"**Status:** {latest['regime']}")
st.markdown(
    f"**Updated:** {latest.name.date()} "
)

delta = latest["prob_up"] - prev["prob_up"]
st.metric(
    "Weekly Signal Change",
    f"{latest['prob_up']:.2f}",
    f"{delta:+.2f}"
)

# ------------------------------------------------------------
# PRICE + RISK CHART
# ------------------------------------------------------------
st.subheader("📈 Price Trend & Risk Signals")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    prob_data.index,
    prob_data["adj_price"],
    color="black",
    linewidth=2,
    label="Real Home Price (Inflation-Adjusted)"
)

for i in range(len(prob_data) - 1):
    r = prob_data["regime"].iloc[i]
    color = (
        "green" if r == "Supportive Market"
        else "gold" if r == "Mixed Signals"
        else "red"
    )
    ax.axvspan(
        prob_data.index[i],
        prob_data.index[i + 1],
        color=color,
        alpha=0.12
    )

ax.set_title("Tampa Housing Market: Price & Risk Signals")
ax.set_ylabel("Inflation-Adjusted Home Price")
ax.legend()
st.pyplot(fig)

# ------------------------------------------------------------
# WEEKLY SIGNAL CHART (LAST 12 WEEKS)
# ------------------------------------------------------------
st.subheader("🕒 Weekly Housing Signal (Last 12 Weeks)")

recent = prob_data.tail(12)

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    recent.index,
    recent["prob_up"],
    marker="o",
    linewidth=2.5,
    color="black"
)

ax2.axhline(0.65, color="green", linestyle="--", alpha=0.6)
ax2.axhline(0.45, color="red", linestyle="--", alpha=0.6)

ax2.set_ylim(0, 1)
ax2.set_ylabel("Market Confidence")
ax2.set_title("Weekly Market Outlook")

fig2.text(
    0.5, -0.25,
    "HOW TO READ THIS CHART\n"
    "• Each dot shows the model’s weekly view of the housing market.\n"
    "• Higher values = more supportive conditions for home prices.\n"
    "• Green zone = supportive market.\n"
    "• Red zone = elevated risk.\n"
    "• Focus on recent direction.",
    ha="center",
    fontsize=10
)

st.pyplot(fig2)
