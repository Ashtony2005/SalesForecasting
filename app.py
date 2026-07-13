"""
Sales Forecasting & Demand Intelligence Dashboard
Streamlit app — run locally with `streamlit run app.py`
or deploy on Streamlit Community Cloud.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ------------------------------------------------------------------
# Data loading & caching
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter

    def season(m):
        if m in [12, 1, 2]:
            return 0
        if m in [3, 4, 5]:
            return 1
        if m in [6, 7, 8]:
            return 2
        return 3

    df["Season"] = df["Order Month"].apply(season)
    return df


def season_code(m):
    if m in [12, 1, 2]:
        return 0
    if m in [3, 4, 5]:
        return 1
    if m in [6, 7, 8]:
        return 2
    return 3


@st.cache_data
def monthly_series(df, category=None, region=None):
    sub = df.copy()
    if category and category != "All":
        sub = sub[sub["Category"] == category]
    if region and region != "All":
        sub = sub[sub["Region"] == region]
    s = sub.set_index("Order Date").resample("MS")["Sales"].sum()
    s.index.freq = "MS"
    return s


def xgb_forecast(series, steps=3, holdout=3):
    """Recursive XGBoost forecast with lag features. Returns forecast + holdout metrics."""
    feat = pd.DataFrame({"Sales": series})
    feat["Lag1"] = feat["Sales"].shift(1)
    feat["Lag2"] = feat["Sales"].shift(2)
    feat["Lag3"] = feat["Sales"].shift(3)
    feat["RollMean3"] = feat["Sales"].shift(1).rolling(3).mean()
    feat["Month"] = feat.index.month
    feat["Quarter"] = feat.index.quarter
    feat["Season"] = feat["Month"].apply(season_code)
    feat = feat.dropna()
    X_cols = ["Lag1", "Lag2", "Lag3", "RollMean3", "Month", "Quarter", "Season"]

    if len(feat) < holdout + 6:
        return None, None, None  # not enough data

    train_feat = feat.iloc[:-holdout]
    test_feat = feat.iloc[-holdout:]

    model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.07,
                          subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(train_feat[X_cols], train_feat["Sales"])

    test_preds = model.predict(test_feat[X_cols])
    mae = float(np.mean(np.abs(test_feat["Sales"].values - test_preds)))
    rmse = float(np.sqrt(np.mean((test_feat["Sales"].values - test_preds) ** 2)))

    # refit on full series, forecast forward recursively
    full_model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.07,
                               subsample=0.9, colsample_bytree=0.9, random_state=42)
    full_model.fit(feat[X_cols], feat["Sales"])
    cur = series.copy()
    last_date = series.index[-1]
    preds = []
    for i in range(1, steps + 1):
        next_date = last_date + pd.DateOffset(months=i)
        lag1, lag2, lag3 = cur.iloc[-1], cur.iloc[-2], cur.iloc[-3]
        rollmean3 = cur.iloc[-3:].mean()
        row = pd.DataFrame([{
            "Lag1": lag1, "Lag2": lag2, "Lag3": lag3, "RollMean3": rollmean3,
            "Month": next_date.month, "Quarter": next_date.quarter,
            "Season": season_code(next_date.month),
        }])
        p = full_model.predict(row[X_cols])[0]
        preds.append(p)
        cur.loc[next_date] = p
    fc = pd.Series(preds, index=pd.date_range(last_date + pd.DateOffset(months=1), periods=steps, freq="MS"))
    return fc, mae, rmse


@st.cache_data
def compute_anomalies(df):
    weekly = df.set_index("Order Date").resample("W")["Sales"].sum()
    iso = IsolationForest(contamination=0.06, random_state=42)
    flags = iso.fit_predict(weekly.values.reshape(-1, 1))
    wdf = pd.DataFrame({"Sales": weekly, "iso_anomaly": flags == -1})
    roll_mean = weekly.rolling(6, center=True, min_periods=3).mean()
    roll_std = weekly.rolling(6, center=True, min_periods=3).std()
    z = (weekly - roll_mean) / roll_std
    wdf["zscore"] = z
    wdf["z_anomaly"] = z.abs() > 2
    return wdf


@st.cache_data
def compute_clusters(df):
    sub_monthly = df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="MS")])["Sales"].sum().reset_index()
    rows = []
    for sub in df["Sub-Category"].unique():
        s = sub_monthly[sub_monthly["Sub-Category"] == sub].set_index("Order Date")["Sales"]
        s = s.reindex(pd.date_range(df["Order Date"].min(), df["Order Date"].max(), freq="MS"), fill_value=0)
        yearly = s.resample("YS").sum()
        yoy_growth = yearly.pct_change().mean() if len(yearly) > 1 else 0
        order_rows = df[df["Sub-Category"] == sub]
        rows.append({
            "Sub-Category": sub, "TotalVolume": s.sum(), "YoYGrowth": yoy_growth,
            "Volatility": s.std(), "AvgOrderValue": order_rows["Sales"].mean(),
        })
    feat_df = pd.DataFrame(rows).set_index("Sub-Category").fillna(0)
    X = feat_df[["TotalVolume", "YoYGrowth", "Volatility", "AvgOrderValue"]].values
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat_df["Cluster"] = km.fit_predict(X_scaled)

    summary = feat_df.groupby("Cluster")[["TotalVolume", "YoYGrowth", "Volatility"]].mean()
    ranked = summary.copy()
    ranked["vol_rank"] = ranked["TotalVolume"].rank(ascending=False)
    ranked["growth_rank"] = ranked["YoYGrowth"].rank(ascending=False)
    ranked["volat_rank"] = ranked["Volatility"].rank(ascending=False)

    def label_cluster(row):
        if row["YoYGrowth"] < 0:
            return "Declining Demand"
        if row["vol_rank"] == 1 and row["volat_rank"] != 1:
            return "High Volume, Stable Demand"
        if row["growth_rank"] <= 2 and row["volat_rank"] <= 2:
            return "High Volatility, Growing Demand"
        if row["growth_rank"] == 1:
            return "Growing Demand"
        return "Low Volume, Stable Demand"

    cluster_labels = ranked.apply(label_cluster, axis=1)
    feat_df["ClusterLabel"] = feat_df["Cluster"].map(cluster_labels)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    feat_df["PC1"], feat_df["PC2"] = X_pca[:, 0], X_pca[:, 1]
    return feat_df


# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------
df = load_data()

st.sidebar.title("📦 Sales Intelligence")
page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"])

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    yearly = df.groupby("Order Year")["Sales"].sum().reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(yearly, x="Order Year", y="Sales", title="Total Sales by Year",
                     color_discrete_sequence=["#2E5C8A"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        monthly = df.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
        fig = px.line(monthly, x="Order Date", y="Sales", title="Monthly Sales Trend", markers=True)
        fig.update_traces(line_color="#2E5C8A")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filter by Region & Category")
    regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    categories = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]

    c3, c4 = st.columns(2)
    with c3:
        by_region = filtered.groupby("Region")["Sales"].sum().reset_index()
        fig = px.bar(by_region, x="Region", y="Sales", title="Sales by Region", color_discrete_sequence=["#27AE60"])
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        by_cat = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig = px.pie(by_cat, names="Category", values="Sales", title="Sales by Category")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts use XGBoost with lag features — the model recommended in the model comparison "
               "(Task 3) for its lower MAE/MAPE among the three approaches tested.")

    dim_type = st.selectbox("Forecast dimension", ["Category", "Region"])
    if dim_type == "Category":
        options = ["All"] + sorted(df["Category"].unique().tolist())
        choice = st.selectbox("Select Category", options)
        series = monthly_series(df, category=choice)
    else:
        options = ["All"] + sorted(df["Region"].unique().tolist())
        choice = st.selectbox("Select Region", options)
        series = monthly_series(df, region=choice)

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    with st.spinner("Training model and generating forecast..."):
        fc, mae, rmse = xgb_forecast(series, steps=horizon, holdout=3)

    if fc is None:
        st.warning("Not enough history for this selection to forecast reliably.")
    else:
        fig = go.Figure()
        tail = series.iloc[-18:]
        fig.add_trace(go.Scatter(x=tail.index, y=tail.values, mode="lines+markers", name="Historical Sales",
                                  line=dict(color="#2E5C8A")))
        fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode="lines+markers", name="Forecast",
                                  line=dict(color="#C0392B", dash="dash")))
        fig.update_layout(title=f"{horizon}-Month Forecast — {dim_type}: {choice}", yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model MAE (holdout)", f"${mae:,.0f}")
        c2.metric("Model RMSE (holdout)", f"${rmse:,.0f}")
        c3.metric("Forecast avg / month", f"${fc.mean():,.0f}")

        st.dataframe(fc.rename("Forecasted Sales").reset_index().rename(columns={"index": "Month"}))

elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Two independent detection methods flagged unusual sales weeks: Isolation Forest "
               "(pattern-based) and a Z-score rule (>2 standard deviations from a 6-week rolling mean).")

    wdf = compute_anomalies(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wdf.index, y=wdf["Sales"], mode="lines", name="Weekly Sales",
                              line=dict(color="#2E5C8A")))
    iso_pts = wdf[wdf["iso_anomaly"]]
    z_pts = wdf[wdf["z_anomaly"]]
    fig.add_trace(go.Scatter(x=iso_pts.index, y=iso_pts["Sales"], mode="markers", name="Isolation Forest anomaly",
                              marker=dict(color="#C0392B", size=10)))
    fig.add_trace(go.Scatter(x=z_pts.index, y=z_pts["Sales"], mode="markers", name="Z-score anomaly",
                              marker=dict(color="#D68910", size=13, symbol="x")))
    fig.update_layout(title="Weekly Sales — Detected Anomalies", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected anomaly weeks")
    anomalies = wdf[wdf["iso_anomaly"] | wdf["z_anomaly"]].copy()
    anomalies["Flagged by"] = anomalies.apply(
        lambda r: ", ".join([m for m, f in [("Isolation Forest", r["iso_anomaly"]), ("Z-score", r["z_anomaly"])] if f]),
        axis=1)
    st.dataframe(anomalies[["Sales", "zscore", "Flagged by"]].round(2).sort_values("Sales", ascending=False))

elif page == "Demand Segments":
    st.title("Product Demand Segments")
    st.caption("Sub-categories grouped by total volume, YoY growth, volatility, and average order value "
               "using K-Means clustering (k=4, chosen via the elbow method).")

    feat_df = compute_clusters(df)
    fig = px.scatter(feat_df.reset_index(), x="PC1", y="PC2", color="ClusterLabel", text="Sub-Category",
                      title="Product Segments (PCA Projection)", size_max=15)
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category → Segment mapping")
    st.dataframe(feat_df[["TotalVolume", "YoYGrowth", "Volatility", "AvgOrderValue", "ClusterLabel"]]
                 .round(1).sort_values("ClusterLabel"))
