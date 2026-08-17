"""Adım 1 — Veriyi tanıma (EDA)."""
import os as _os, pathlib as _p
KOK = _p.Path(__file__).resolve().parent.parent
_os.chdir(KOK)
for _d in ("outputs", "figures"):
    (KOK / _d).mkdir(exist_ok=True)

import pandas as pd

pd.set_option("display.width", 200)
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("=== ŞEKİL ===")
print(df.shape)

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== EKSİK DEĞER ===")
print(df.isna().sum()[lambda s: s > 0])

print("\n=== TotalCharges boşluk kontrolü ===")
bos = df["TotalCharges"].astype(str).str.strip() == ""
print("boş string sayısı:", bos.sum())
print(df.loc[bos, ["tenure", "MonthlyCharges", "TotalCharges", "Churn", "Contract"]])

print("\n=== HEDEF DAĞILIMI ===")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).round(4))

print("\n=== TEKİL MÜŞTERİ ===")
print("customerID benzersiz mi:", df["customerID"].is_unique)

print("\n=== SAYISAL ÖZET ===")
df2 = df.copy()
df2["TotalCharges"] = pd.to_numeric(df2["TotalCharges"], errors="coerce")
print(df2[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))

print("\n=== KATEGORİK KIRILIMLARDA CHURN ORANI ===")
kat = [c for c in df.columns if df[c].dtype == object and c not in ("customerID", "Churn")]
df2["churn_bin"] = (df2["Churn"] == "Yes").astype(int)
for c in kat:
    t = df2.groupby(c)["churn_bin"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    t["mean"] = (t["mean"] * 100).round(1)
    print(f"\n--- {c} ---")
    print(t)

print("\n=== TENURE KOVALARINDA CHURN ===")
df2["tenure_kova"] = pd.cut(df2["tenure"], [-1, 6, 12, 24, 48, 72],
                            labels=["0-6 ay", "7-12 ay", "13-24 ay", "25-48 ay", "49-72 ay"])
t = df2.groupby("tenure_kova", observed=True)["churn_bin"].agg(["mean", "size"])
t["mean"] = (t["mean"] * 100).round(1)
print(t)

print("\n=== SÖZLEŞME x TENURE ===")
pt = df2.pivot_table(index="tenure_kova", columns="Contract", values="churn_bin",
                     aggfunc="mean", observed=True)
print((pt * 100).round(1))

print("\n=== AYLIK ÜCRET: churn eden vs etmeyen ===")
print(df2.groupby("Churn")[["MonthlyCharges", "tenure", "TotalCharges"]].mean().round(2))

# Kaybedilen gelir büyüklüğü
churn_musteri = df2[df2["churn_bin"] == 1]
print("\nAyrılan müşterilerin aylık toplam geliri: {:,.0f} $".format(churn_musteri["MonthlyCharges"].sum()))
print("Tüm müşterilerin aylık toplam geliri:     {:,.0f} $".format(df2["MonthlyCharges"].sum()))
print("Gelirin yüzde kaçı risk altında: {:.1f}%".format(
    100 * churn_musteri["MonthlyCharges"].sum() / df2["MonthlyCharges"].sum()))
