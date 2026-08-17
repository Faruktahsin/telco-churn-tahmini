"""Adım 5 — Yazıdaki her sayıyı bağımsız olarak yeniden hesapla."""
import os as _os, pathlib as _p
KOK = _p.Path(__file__).resolve().parent.parent
_os.chdir(KOK)
for _d in ("outputs", "figures"):
    (KOK / _d).mkdir(exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["c"] = (df["Churn"] == "Yes").astype(int)
y = np.load("outputs/y_test.npy")
p_lr = np.load("outputs/olasilik_lr.npy")
p_rf = np.load("outputs/olasilik_rf.npy")

kontrol = []


def esit(ad, iddia, gercek, tol=0.006):
    ok = abs(iddia - gercek) <= tol
    kontrol.append((ad, iddia, round(float(gercek), 4), "OK" if ok else "!!! HATA"))


esit("satır sayısı", 7043, len(df), 0)
esit("sütun sayısı", 21, df.shape[1] - 2, 0)
esit("churn oranı %", 26.5, df.c.mean() * 100, 0.05)
esit("boş TotalCharges", 11, df.TotalCharges_num.isna().sum(), 0)
esit("toplam aylık gelir", 456117, df.MonthlyCharges.sum(), 1)
esit("churn aylık gelir", 139131, df[df.c == 1].MonthlyCharges.sum(), 1)
esit("gelirin risk %'si", 30.5, 100 * df[df.c == 1].MonthlyCharges.sum() / df.MonthlyCharges.sum(), 0.05)

esit("aydan aya %", 42.7, df[df.Contract == "Month-to-month"].c.mean() * 100, 0.05)
esit("1 yıllık %", 11.3, df[df.Contract == "One year"].c.mean() * 100, 0.05)
esit("2 yıllık %", 2.8, df[df.Contract == "Two year"].c.mean() * 100, 0.05)
esit("0-6 ay %", 52.9, df[df.tenure <= 6].c.mean() * 100, 0.05)
esit("49-72 ay %", 9.5, df[df.tenure >= 49].c.mean() * 100, 0.05)
esit("fiber %", 41.9, df[df.InternetService == "Fiber optic"].c.mean() * 100, 0.05)
esit("DSL %", 19.0, df[df.InternetService == "DSL"].c.mean() * 100, 0.05)
esit("elektronik çek %", 45.3, df[df.PaymentMethod == "Electronic check"].c.mean() * 100, 0.05)
esit("kredi kartı oto %", 15.2, df[df.PaymentMethod == "Credit card (automatic)"].c.mean() * 100, 0.05)
esit("kadın %", 26.9, df[df.gender == "Female"].c.mean() * 100, 0.05)
esit("erkek %", 26.2, df[df.gender == "Male"].c.mean() * 100, 0.05)
esit("churn ort aylık ücret", 74.44, df[df.c == 1].MonthlyCharges.mean(), 0.01)
esit("kalan ort aylık ücret", 61.27, df[df.c == 0].MonthlyCharges.mean(), 0.01)
esit("tenure-TotalCharges kor.", 0.826, df[["tenure", "TotalCharges_num"]].corr().iloc[0, 1], 0.001)

esit("test seti n", 1409, len(y), 0)
esit("eğitim seti n", 5634, 7043 - len(y), 0)
esit("kukla accuracy", 0.7346, 1 - y.mean(), 0.001)
esit("LR test ROC-AUC", 0.8417, roc_auc_score(y, p_lr), 0.0005)
esit("RF test ROC-AUC", 0.8455, roc_auc_score(y, p_rf), 0.0005)
esit("RF test PR-AUC", 0.6580, average_precision_score(y, p_rf), 0.0005)
esit("LR test PR-AUC", 0.6331, average_precision_score(y, p_lr), 0.0005)
esit("AUC farkı", 0.0038, roc_auc_score(y, p_rf) - roc_auc_score(y, p_lr), 0.0005)

tn, fp, fn, tp = confusion_matrix(y, (p_rf >= 0.50).astype(int)).ravel()
esit("RF@0.50 kaçan", 82, fn, 0)
esit("RF@0.50 recall", 0.781, tp / (tp + fn), 0.001)
esit("RF@0.50 precision", 0.529, tp / (tp + fp), 0.001)
m50 = fn * 893 + fp * 120
esit("RF@0.50 maliyet", 104426, m50, 1)

tn, fp, fn, tp = confusion_matrix(y, (p_rf >= 0.23).astype(int)).ravel()
esit("RF@0.23 kaçan", 19, fn, 0)
esit("RF@0.23 recall", 0.949, tp / (tp + fn), 0.001)
esit("RF@0.23 precision", 0.394, tp / (tp + fp), 0.001)
m23 = fn * 893 + fp * 120
esit("RF@0.23 maliyet", 82487, m23, 1)
esit("tasarruf", 21939, m50 - m23, 1)
esit("kurtarılan müşteri", 63, 82 - 19, 0)

tn, fp, fn, tp = confusion_matrix(y, (p_lr >= 0.50).astype(int)).ravel()
esit("LR@0.50 recall", 0.783, tp / (tp + fn), 0.001)
esit("LR@0.50 precision", 0.505, tp / (tp + fp), 0.001)

d = pd.DataFrame({"y": y, "p": p_rf})
d["des"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
g = d.groupby("des", observed=True).y.agg(["sum", "size", "mean"])
esit("D1 churn %", 75.9, g.loc[1, "mean"] * 100, 0.05)
esit("D1 lift", 2.86, g.loc[1, "mean"] / y.mean(), 0.005)
esit("üst %20 yakalama", 50.5, g.loc[:2, "sum"].sum() / y.sum() * 100, 0.05)
esit("üst %30 yakalama", 66.6, g.loc[:3, "sum"].sum() / y.sum() * 100, 0.05)
esit("D9+D10 churn adet", 3, g.loc[9:10, "sum"].sum(), 0)
esit("D9+D10 müşteri", 282, g.loc[9:10, "size"].sum(), 0)

k = pd.read_csv("outputs/katsayilar.csv").set_index("degisken")
esit("fiber odds", 3.41, k.loc["kat__InternetService_Fiber optic", "odds_orani"], 0.005)
esit("tenure odds", 0.31, k.loc["say__tenure", "odds_orani"], 0.005)
esit("2 yıl odds", 0.25, k.loc["kat__Contract_Two year", "odds_orani"], 0.005)
esit("1 yıl odds", 0.49, k.loc["kat__Contract_One year", "odds_orani"], 0.005)
esit("MonthlyCharges kats.", -0.555, k.loc["say__MonthlyCharges", "katsayi"], 0.001)
esit("MonthlyCharges odds", 0.574, k.loc["say__MonthlyCharges", "odds_orani"], 0.001)

pi = pd.read_csv("outputs/permutasyon_onem.csv").set_index("degisken")
esit("tenure permütasyon", 0.041, pi.loc["tenure", "onem"], 0.0006)
esit("Contract permütasyon", 0.036, pi.loc["Contract", "onem"], 0.0006)
esit("ilk4 dışı toplam", 0.010, pi.onem.iloc[4:].sum(), 0.010)

cvd = pd.read_csv("outputs/cv_sonuclari.csv").set_index("model")
esit("CV LR roc_auc", 0.846, cvd.loc["Lojistik Regresyon", "roc_auc"], 0.0006)
esit("CV LR recall", 0.543, cvd.loc["Lojistik Regresyon", "recall"], 0.0006)
esit("CV LR-dengeli recall", 0.802, cvd.loc["Lojistik Regresyon (dengeli)", "recall"], 0.0006)
esit("CV RF ham roc_auc", 0.826, cvd.loc["Random Forest", "roc_auc"], 0.0006)
esit("CV RF-dengeli roc_auc", 0.845, cvd.loc["Random Forest (dengeli)", "roc_auc"], 0.0006)

out = pd.DataFrame(kontrol, columns=["iddia", "yazıdaki", "hesaplanan", "durum"])
print(out.to_string(index=False))
print("\nTOPLAM:", len(out), "| HATA:", (out.durum != "OK").sum())
