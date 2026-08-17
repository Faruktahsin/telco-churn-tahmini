"""Adım 3 — Eşik duyarlılığı ve desil (lift) analizi."""
import os as _os, pathlib as _p
KOK = _p.Path(__file__).resolve().parent.parent
_os.chdir(KOK)
for _d in ("outputs", "figures"):
    (KOK / _d).mkdir(exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

y = np.load("outputs/y_test.npy")
p_lr = np.load("outputs/olasilik_lr.npy")
p_rf = np.load("outputs/olasilik_rf.npy")

print("=========== MALİYET ORANINA GÖRE EN İYİ EŞİK (Random Forest) ===========")
print("Oran = kaçırılan churn maliyeti / gereksiz teklif maliyeti\n")
esikler = np.linspace(0.05, 0.95, 91)
satirlar = []
for oran in [2, 3, 5, 7.4, 10, 15, 20]:
    KAYIP, TEKLIF = oran, 1.0
    m = []
    for e in esikler:
        tn, fp, fn, tp = confusion_matrix(y, (p_rf >= e).astype(int)).ravel()
        m.append(fn * KAYIP + fp * TEKLIF)
    m = np.array(m)
    e_iyi = esikler[m.argmin()]
    tn, fp, fn, tp = confusion_matrix(y, (p_rf >= e_iyi).astype(int)).ravel()
    satirlar.append({"maliyet_orani": f"1:{oran:g}", "en_iyi_esik": round(e_iyi, 2),
                     "recall": round(tp / (tp + fn), 3),
                     "precision": round(tp / (tp + fp), 3),
                     "temas_edilen_musteri": tp + fp})
print(pd.DataFrame(satirlar).to_string(index=False))

print("\n=========== DESİL ANALİZİ (Random Forest, test seti) ===========")
d = pd.DataFrame({"y": y, "p": p_rf})
d["desil"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10,
                     labels=[f"D{i}" for i in range(1, 11)])
t = d.groupby("desil", observed=True).agg(
    musteri=("y", "size"), gercek_churn=("y", "sum"), churn_orani=("y", "mean"),
    ort_olasilik=("p", "mean")).reset_index()
t["churn_orani"] = (t["churn_orani"] * 100).round(1)
t["ort_olasilik"] = t["ort_olasilik"].round(3)
t["kumulatif_yakalanan_%"] = (t["gercek_churn"].cumsum() / y.sum() * 100).round(1)
t["lift"] = (t["gercek_churn"] / t["musteri"] / y.mean()).round(2)
print(t.to_string(index=False))

print(f"\nTest setinde toplam gerçek churn: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")
print(f"En riskli %20'ye bakarak churn'lerin %{t['kumulatif_yakalanan_%'].iloc[1]:.0f}'ini yakalıyoruz.")
print(f"En riskli %30'a bakarak churn'lerin %{t['kumulatif_yakalanan_%'].iloc[2]:.0f}'ini yakalıyoruz.")

print("\n=========== MODEL KARŞILAŞTIRMASI: en riskli %20 ===========")
for ad, p in [("Lojistik Regresyon", p_lr), ("Random Forest", p_rf)]:
    k = int(len(y) * 0.20)
    idx = np.argsort(-p)[:k]
    print(f"{ad:22s} -> {y[idx].sum():3d}/{y.sum()} churn yakalandı "
          f"({y[idx].sum()/y.sum()*100:.1f}%), bu dilimde churn oranı %{y[idx].mean()*100:.1f}")
