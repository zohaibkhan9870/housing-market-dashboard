import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Tampa Housing Market Risk Dashboard",
    layout="centered"
)

st.title("🏠 Tampa Housing Market Risk Dashboard")

st.markdown("""
This dashboard shows a **weekly housing market signal for Tampa, Florida**  
designed for real estate investors and housing professionals.
""")

# --------------------------------
# LOAD FRED DATA (NO pandas_datareader)
# --------------------------------
def load_fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")

with st.spinner("Loading economic data..."):
    mortgage = load_fred("MORTGAGE30US")
    vacancy = load_fred("RRVRUSQ156N")
    cpi = load_fred("CPIAUCSL")

fed_data = mortgage.join(vacancy).join(cpi)
fed_data.columns = ["interest", "vacancy", "cpi"]
fed_data = fed_data.ffill().dropna()

# --------------------------------
# LOAD ZILLOW FILES (REPO FILES)
# --------------------------------
zillow_price = pd.read_csv(
    "Metro_median_sale_price_uc_sfrcondo_sm_week.csv"
)
zillow_value = pd.read_csv(
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)

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

metro_name = price_matches["RegionName"].values[0]

price = pd.DataFrame(price_matches.iloc[0, 5:])
value = pd.DataFrame(value_matches.iloc[0, 5:])

# --------------------------------
# PREP DATA
# --------------------------------
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

# --------------------------------
# FEATURE ENGINEERING
# --------------------------------
price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100

price_data["next_quarter"] = price_data["adj_price"].shift(-13)
price_data["change"] = (
    price_data["next_quarter"] > price_data["adj_price"]
).astype(int)

price_data["price_13w_change"] = price_data["adj_price"].pct_change(13)
price_data["value_52w_change"] = price_data["adj_value"].pct_change(52)

price_data.dropna(inplace=True)

# --------------------------------
# MODEL
# --------------------------------
predictors = [
    "adj_price",
    "adj_value",
    "interest",
    "price_13w_change",
    "value_52w_change"
]

STEP = 52
START = max(104, int(price_data.shape[0] * 0.5))

def predict_proba(train, test):
    rf = RandomForestClassifier(
        min_samples_split=10,
        random_state=1
    )
    rf.fit(train[predictors], train["change"])
    return rf.predict_proba(test[predictors])[:, 1]

all_probs = []

for i in range(START, price_data.shape[0], STEP):
    train = price_data.iloc[:i]
    test = price_data.iloc[i:i+STEP]
    all_probs.append(predict_proba(train, test))

probs = np.concatenate(all_probs)

prob_data = price_data.iloc[START:].copy()
prob_data["prob_up"] = probs

def label_regime(p):
    if p > 0.65:
        return "Supportive"
    elif p < 0.45:
        return "High Risk"
    else:
        return "Mixed"

prob_data["regime"] = prob_data["prob_up"].apply(label_regime)

# --------------------------------
# CURRENT SIGNAL
# --------------------------------
latest = prob_data.iloc[-1]
previous = prob_data.iloc[-2]

st.subheader("📍 Current Market Environment")
st.caption(f"Metro: {metro_name} | Updated: {datetime.today().date()}")

if latest["regime"] == "Supportive":
    st.success("🟢 Supportive Market")
elif latest["regime"] == "Mixed":
    st.warning("🟡 Mixed Signals")
else:
    st.error("🔴 High Risk / Caution")

# --------------------------------
# WEEK-OVER-WEEK CHANGE
# --------------------------------
st.subheader("📈 Weekly Signal Change")

if latest["regime"] == previous["regime"]:
    st.info(f"➡️ Signal unchanged: {latest['regime']}")
else:
    st.warning(f"🔄 Signal changed from {previous['regime']} to {latest['regime']}")

# --------------------------------
# PRICE + RISK CHART
# --------------------------------
st.subheader("📊 Tampa Home Prices & Risk Signals")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    prob_data.index,
    prob_data["adj_price"],
    color="black",
    label="Real Home Price"
)

for i in range(len(prob_data) - 1):
    regime = prob_data["regime"].iloc[i]
    color = (
        "green" if regime == "Supportive"
        else "yellow" if regime == "Mixed"
        else "red"
    )
    ax.axvspan(
        prob_data.index[i],
        prob_data.index[i+1],
        color=color,
        alpha=0.12
    )

ax.set_title("Tampa Housing Market: Price Trend & Risk Signals")
ax.set_ylabel("Inflation-Adjusted Home Price")
ax.legend()

st.pyplot(fig)

st.markdown("""
**How to read this chart:**
- **Black line** = Typical Tampa home price (inflation-adjusted)
- **Green background** = Supportive market
- **Yellow background** = Mixed signals
- **Red background** = Elevated risk

Focus on **recent weeks**, not old history.  
This signal updates **weekly**.
""")
