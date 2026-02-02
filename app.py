
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config("Bank Fraud Investigation", layout="wide")
st.title("🏦 Fraud Detection Investigation Dashboard")

file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📥 Uploaded Data Preview")
    st.dataframe(df.head())

    # ================= AUTO DETECT ACCOUNT COLUMN =================
    possible_account_cols = ["AccountID", "account_id", "Account Id", "ACCOUNTID", "accountid"]
    account_col = None
    for c in possible_account_cols:
        if c in df.columns:
            account_col = c
            break

    if account_col is None:
        st.error("❌ Account column not found. Rename your column to AccountID")
        st.stop()

    st.success(f"✅ Using Account Column: {account_col}")

    # ================= FEATURE ENGINEERING =================
    agg = df.groupby(account_col).agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        unique_targets=("TransactionID", "nunique"),
    ).reset_index().rename(columns={account_col: "AccountID"})

    # ================= ANOMALY MODEL =================
    X = agg[["total_amount", "avg_amount", "unique_targets"]]
    model = IsolationForest(contamination=0.1, random_state=42)
    agg["is_anomaly"] = model.fit_predict(X)
    agg["is_anomaly"] = agg["is_anomaly"].map({1: 0, -1: 1})

    # ================= FRAUD PROBABILITY =================
    agg["fraud_probability"] = np.clip(
        0.3 * agg["is_anomaly"]
        + 0.4 * (agg["total_amount"] / agg["total_amount"].max())
        + 0.3 * (agg["avg_amount"] / agg["avg_amount"].max()),
        0,
        1,
    )

    # ================= CONTRIBUTION =================
    baseline = X.median()
    dev = abs(X - baseline)
    contrib = dev.div(dev.sum(axis=1), axis=0) * 100
    contrib.columns = ["total_amount_contrib", "avg_amount_contrib", "unique_targets_contrib"]
    agg = pd.concat([agg, contrib], axis=1)

    # ================= RISK LABEL =================
    def risk_label(p):
        if p > 0.75:
            return "HIGH"
        elif p > 0.4:
            return "MEDIUM"
        return "LOW"

    agg["risk_label"] = agg["fraud_probability"].apply(risk_label)

    # ================= FRAUD REASONS =================
    def explain(row):
        reasons = []
        if row["total_amount_contrib"] > 50:
            reasons.append("High total transaction volume")
        if row["avg_amount_contrib"] > 30:
            reasons.append("Unusual large transactions")
        if row["unique_targets_contrib"] > 30:
            reasons.append("Many transaction targets")
        if row["is_anomaly"] == 1:
            reasons.append("Statistical anomaly detected")
        return ", ".join(reasons)

    agg["fraud_reasons"] = agg.apply(explain, axis=1)

    # ================= ACCOUNT DETAILS =================
    details_cols = ["Email", "PhoneNumber", "IPAddress", "DeviceID"]
    available_cols = [c for c in details_cols if c in df.columns]

    details = df.groupby(account_col).first()[available_cols].reset_index()
    details = details.rename(columns={account_col: "AccountID"})
    agg = agg.merge(details, on="AccountID", how="left")

    # ================= SEARCH =================
    search = st.text_input("🔍 Search AccountID (example: ACC_5915)")
    if search:
        agg = agg[agg["AccountID"].str.contains(search)]

    # ================= FINAL TABLE =================
    final_cols = [
        "AccountID",
        "fraud_probability",
        "risk_label",
        "total_amount_contrib",
        "avg_amount_contrib",
        "unique_targets_contrib",
        "fraud_reasons",
    ] + available_cols

    final_table = agg[final_cols].sort_values("fraud_probability", ascending=False)

    st.subheader("🚨 FRAUD INVESTIGATION TABLE (BANK STYLE)")
    st.dataframe(final_table)

    # ================= DOWNLOAD =================
    st.download_button(
        "⬇ Download Report CSV", final_table.to_csv(index=False), "fraud_report.csv"
    )

