"""Adım 2 — Modelleme, çapraz doğrulama, eşik optimizasyonu."""
import os as _os, pathlib as _p
KOK = _p.Path(__file__).resolve().parent.parent
_os.chdir(KOK)
for _d in ("outputs", "figures"):
    (KOK / _d).mkdir(exist_ok=True)

import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RASTGELE = 42
np.random.seed(RASTGELE)

# ---------------------------------------------------------------- veri hazırlığı
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("TotalCharges NaN sayısı:", df["TotalCharges"].isna().sum())
print("Bu satırların tenure değerleri:", sorted(df.loc[df["TotalCharges"].isna(), "tenure"].unique()))
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

df = df.drop(columns=["customerID"])
y = (df.pop("Churn") == "Yes").astype(int)
X = df

sayisal = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
kategorik = [c for c in X.columns if c not in sayisal]
print(f"\nSayısal: {len(sayisal)} | Kategorik: {len(kategorik)}")

print("\n--- Sayısal değişkenler arası korelasyon ---")
print(X[["tenure", "MonthlyCharges", "TotalCharges"]].corr().round(3))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RASTGELE)
print(f"\nEğitim: {X_tr.shape[0]} | Test: {X_te.shape[0]}")
print(f"Eğitimde churn oranı: {y_tr.mean():.4f} | Testte: {y_te.mean():.4f}")

on_isleme = ColumnTransformer([
    ("say", StandardScaler(), sayisal),
    ("kat", OneHotEncoder(drop="first", handle_unknown="ignore"), kategorik),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RASTGELE)
skor = ["roc_auc", "average_precision", "recall", "precision", "f1", "accuracy"]

modeller = {
    "Kukla (hep 'ayrılmaz')": Pipeline([
        ("on", on_isleme), ("mdl", DummyClassifier(strategy="most_frequent"))]),
    "Lojistik Regresyon": Pipeline([
        ("on", on_isleme),
        ("mdl", LogisticRegression(max_iter=2000, random_state=RASTGELE))]),
    "Lojistik Regresyon (dengeli)": Pipeline([
        ("on", on_isleme),
        ("mdl", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RASTGELE))]),
    "Random Forest": Pipeline([
        ("on", on_isleme),
        ("mdl", RandomForestClassifier(n_estimators=400, random_state=RASTGELE, n_jobs=-1))]),
    "Random Forest (dengeli)": Pipeline([
        ("on", on_isleme),
        ("mdl", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                       min_samples_leaf=5, random_state=RASTGELE, n_jobs=-1))]),
}

print("\n=========== 5 KATLI ÇAPRAZ DOĞRULAMA (eğitim seti) ===========")
cv_tablo = []
for ad, pipe in modeller.items():
    r = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=skor, n_jobs=-1)
    satir = {"model": ad}
    for s in skor:
        satir[s] = r[f"test_{s}"].mean()
        satir[s + "_std"] = r[f"test_{s}"].std()
    cv_tablo.append(satir)
cv_df = pd.DataFrame(cv_tablo).set_index("model")
print(cv_df[skor].round(4).to_string())
print("\nStandart sapmalar (ROC-AUC):")
print(cv_df["roc_auc_std"].round(4).to_string())

# ---------------------------------------------------------------- hiperparametre
print("\n=========== RANDOM FOREST HİPERPARAMETRE ARAMASI ===========")
grid = GridSearchCV(
    Pipeline([("on", on_isleme),
              ("mdl", RandomForestClassifier(class_weight="balanced_subsample",
                                             random_state=RASTGELE, n_jobs=-1))]),
    {"mdl__n_estimators": [300, 600],
     "mdl__max_depth": [6, 10, None],
     "mdl__min_samples_leaf": [1, 5, 15]},
    scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_tr, y_tr)
print("En iyi parametreler:", grid.best_params_)
print(f"En iyi CV ROC-AUC: {grid.best_score_:.4f}")

# ---------------------------------------------------------------- test seti
lr = modeller["Lojistik Regresyon (dengeli)"].fit(X_tr, y_tr)
rf = grid.best_estimator_

print("\n=========== TEST SETİ SONUÇLARI ===========")
sonuc = {}
for ad, m in [("Lojistik Regresyon (dengeli)", lr), ("Random Forest (ayarlı)", rf)]:
    p = m.predict_proba(X_te)[:, 1]
    print(f"\n### {ad}")
    print(f"ROC-AUC : {roc_auc_score(y_te, p):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_te, p):.4f}")
    print("\n0.50 eşiği ile:")
    tahmin = (p >= 0.5).astype(int)
    print(classification_report(y_te, tahmin, target_names=["Kalır", "Ayrılır"], digits=3))
    print("Karmaşıklık matrisi [satır=gerçek, sütun=tahmin]:")
    print(confusion_matrix(y_te, tahmin))
    sonuc[ad] = p

# ---------------------------------------------------------------- eşik / maliyet
print("\n=========== EŞİK OPTİMİZASYONU (iş maliyeti) ===========")
# Varsayım: ayrılan müşterinin ortalama aylık geliri ~74.4$, 12 aylık kayıp ≈ 893$
# Elde tutma teklifi maliyeti ≈ 120$ (2 ay %50 indirim ~ 74$ + operasyon)
KAYIP = 893.0     # kaçırılan churn'ün maliyeti (FN)
TEKLIF = 120.0    # gereksiz teklifin maliyeti (FP)

for ad, p in sonuc.items():
    esikler = np.linspace(0.05, 0.95, 91)
    maliyetler = []
    for e in esikler:
        t = (p >= e).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, t).ravel()
        maliyetler.append(fn * KAYIP + fp * TEKLIF)
    maliyetler = np.array(maliyetler)
    en_iyi = esikler[maliyetler.argmin()]
    t = (p >= en_iyi).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, t).ravel()
    varsayilan = (p >= 0.5).astype(int)
    tn0, fp0, fn0, tp0 = confusion_matrix(y_te, varsayilan).ravel()
    print(f"\n### {ad}")
    print(f"En iyi eşik: {en_iyi:.2f} | maliyet: {maliyetler.min():,.0f}$ "
          f"(0.50 eşiğinde: {(fn0*KAYIP+fp0*TEKLIF):,.0f}$)")
    print(f"Bu eşikte  -> TP:{tp} FP:{fp} FN:{fn} TN:{tn} | "
          f"recall={tp/(tp+fn):.3f} precision={tp/(tp+fp):.3f}")
    print(f"0.50'de    -> TP:{tp0} FP:{fp0} FN:{fn0} TN:{tn0} | "
          f"recall={tp0/(tp0+fn0):.3f} precision={tp0/(tp0+fp0):.3f}")
    print(f"Tasarruf: {(fn0*KAYIP+fp0*TEKLIF) - maliyetler.min():,.0f}$")

# ---------------------------------------------------------------- yorumlanabilirlik
print("\n=========== LOJİSTİK REGRESYON KATSAYILARI (odds oranı) ===========")
isimler = lr.named_steps["on"].get_feature_names_out()
kats = lr.named_steps["mdl"].coef_[0]
kdf = pd.DataFrame({"degisken": isimler, "katsayi": kats})
kdf["odds_orani"] = np.exp(kdf["katsayi"])
kdf = kdf.sort_values("katsayi", ascending=False)
print("\n-- Churn olasılığını EN ÇOK ARTIRAN 10 --")
print(kdf.head(10).round(3).to_string(index=False))
print("\n-- Churn olasılığını EN ÇOK AZALTAN 10 --")
print(kdf.tail(10).round(3).to_string(index=False))

print("\n=========== RANDOM FOREST PERMÜTASYON ÖNEMİ (test seti, ROC-AUC) ===========")
pi = permutation_importance(rf, X_te, y_te, n_repeats=15, random_state=RASTGELE,
                            scoring="roc_auc", n_jobs=-1)
pidf = pd.DataFrame({"degisken": X_te.columns,
                     "onem": pi.importances_mean,
                     "std": pi.importances_std}).sort_values("onem", ascending=False)
print(pidf.head(12).round(4).to_string(index=False))

# ---------------------------------------------------------------- kaydet
np.save("outputs/olasilik_lr.npy", sonuc["Lojistik Regresyon (dengeli)"])
np.save("outputs/olasilik_rf.npy", sonuc["Random Forest (ayarlı)"])
np.save("outputs/y_test.npy", y_te.values)
kdf.to_csv("outputs/katsayilar.csv", index=False)
pidf.to_csv("outputs/permutasyon_onem.csv", index=False)
cv_df.to_csv("outputs/cv_sonuclari.csv")
with open("outputs/en_iyi_parametreler.json", "w") as f:
    json.dump({k: str(v) for k, v in grid.best_params_.items()}, f, indent=2)
print("\nÇıktılar kaydedildi.")
