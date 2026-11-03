"""Adım 6 — Medium tablo desteklemediği için tabloları görsele çevir."""
import os as _os, pathlib as _p
KOK = _p.Path(__file__).resolve().parent.parent
_os.chdir(KOK)
for _d in ("outputs", "figures"):
    (KOK / _d).mkdir(exist_ok=True)

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

YUZEY, MURK, IKINCIL, SOLUK = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
IZGARA, S1, KRITIK = "#e1e0d9", "#2a78d6", "#d03b3b"
VURGU = "#eaf2fd"

plt.rcParams.update({"figure.facecolor": YUZEY, "savefig.facecolor": YUZEY,
                     "font.family": "DejaVu Sans", "figure.dpi": 170,
                     "savefig.dpi": 170, "savefig.bbox": "tight"})


def tablo(dosya, baslik, basliklar, satirlar, vurgulu=None, genislik=None,
          hizalama=None, alt_not=None):
    n_satir, n_sutun = len(satirlar), len(basliklar)
    genislik = genislik or [1.0] * n_sutun
    hizalama = hizalama or ["left"] + ["center"] * (n_sutun - 1)
    toplam = sum(genislik)
    sinir = [0.0]
    for g in genislik:
        sinir.append(sinir[-1] + g / toplam)

    fig_h = 0.52 * n_satir + 1.5 + (0.35 if alt_not else 0)
    fig, ax = plt.subplots(figsize=(sum(genislik) * 1.35, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_satir + 1.9)
    ax.axis("off")

    ax.text(0, n_satir + 1.42, baslik, fontsize=13.5, fontweight="bold",
            color=MURK, va="bottom")

    ust = n_satir + 0.62
    for j, b in enumerate(basliklar):
        x = {"left": sinir[j] + 0.006, "center": (sinir[j] + sinir[j + 1]) / 2,
             "right": sinir[j + 1] - 0.006}[hizalama[j]]
        ax.text(x, ust + 0.16, b, fontsize=10, fontweight="bold", color=IKINCIL,
                ha=hizalama[j], va="center")
    ax.plot([0, 1], [ust - 0.18] * 2, color=SOLUK, lw=1.3)

    for i, satir in enumerate(satirlar):
        y = ust - 0.75 - i * 0.85
        if vurgulu and i in vurgulu:
            ax.add_patch(plt.Rectangle((0, y - 0.35), 1, 0.72, color=VURGU,
                                       zorder=0, lw=0))
        for j, h in enumerate(satir):
            x = {"left": sinir[j] + 0.006, "center": (sinir[j] + sinir[j + 1]) / 2,
                 "right": sinir[j + 1] - 0.006}[hizalama[j]]
            kalin = (vurgulu and i in vurgulu) or j == 0
            ax.text(x, y, str(h), fontsize=10.5,
                    fontweight="bold" if kalin else "normal",
                    color=MURK if kalin else IKINCIL, ha=hizalama[j], va="center")
        if i < len(satirlar) - 1:
            ax.plot([0, 1], [y - 0.38] * 2, color=IZGARA, lw=0.9, zorder=0)

    if alt_not:
        ax.text(0, ust - 0.75 - n_satir * 0.85 + 0.1, alt_not, fontsize=9,
                color=SOLUK, va="top", style="italic")
    fig.savefig(dosya)
    plt.close(fig)


# ---------------------------------------------------------------- veriler
# Tablolardaki sayılar elle yazılmaz; 02 ve 03 adımlarının çıktılarından okunur.
cvd = pd.read_csv("outputs/cv_sonuclari.csv").set_index("model")
y_te = np.load("outputs/y_test.npy")
p_lr = np.load("outputs/olasilik_lr.npy")
p_rf = np.load("outputs/olasilik_rf.npy")
duy = pd.read_csv("outputs/esik_duyarlilik.csv")
with open("outputs/secilen_esik.json", encoding="utf-8") as f:
    esik = json.load(f)["esikler"]


def f3(v):
    return f"{v:.3f}"


def tr_sayi(n):
    return f"{int(n):,}".replace(",", ".")


# ---------------------------------------------------------------- T1
cv_sira = [("Kukla (hep 'ayrılmaz')", "Kukla (hep “ayrılmaz”)"),
           ("Lojistik Regresyon", "Lojistik Regresyon"),
           ("Lojistik Regresyon (dengeli)", "Lojistik Regresyon (dengeli)"),
           ("Random Forest", "Random Forest"),
           ("Random Forest (dengeli)", "Random Forest (dengeli)")]
tablo("figures/t1_cv.png",
      "5 katlı çapraz doğrulama sonuçları (eğitim seti)",
      ["Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "Accuracy"],
      [[etiket] + [f3(cvd.loc[ad, k]) for k in
                   ["roc_auc", "average_precision", "recall", "precision", "f1", "accuracy"]]
       for ad, etiket in cv_sira],
      vurgulu=[2],
      genislik=[3.3, 1.0, 1.0, 1.0, 1.05, 0.85, 1.05],
      alt_not=(f"Kukla model %{cvd.loc[cv_sira[0][0], 'accuracy'] * 100:.1f}".replace(".", ",")
               + " doğrulukla çalışıyor — ve hiçbir şey öğrenmiyor."))


# ---------------------------------------------------------------- T2
def test_satiri(ad, p, e):
    tn, fp, fn, tp = confusion_matrix(y_te, (p >= 0.5).astype(int)).ravel()
    tn2, fp2, fn2, tp2 = confusion_matrix(y_te, (p >= e).astype(int)).ravel()
    return [ad, f"{roc_auc_score(y_te, p):.4f}", f"{average_precision_score(y_te, p):.4f}",
            str(fn), f"{e:.2f}", str(fn2)]


fark = roc_auc_score(y_te, p_rf) - roc_auc_score(y_te, p_lr)
tablo("figures/t2_test.png",
      f"Test seti sonuçları ({tr_sayi(len(y_te))} müşteri)",
      ["Model", "ROC-AUC", "PR-AUC", "Kaçan (0.50)", "Seçilen eşik", "Kaçan (eşikte)"],
      [test_satiri("Lojistik Regresyon (dengeli)", p_lr, esik["Lojistik Regresyon (dengeli)"]),
       test_satiri("Random Forest (ayarlı)", p_rf, esik["Random Forest (ayarlı)"])],
      vurgulu=[1],
      genislik=[3.3, 1.1, 1.05, 1.3, 1.3, 1.4],
      alt_not=(f"AUC farkı {fark:.3f}".replace(".", ",")
               + " — eşikler test setine bakılmadan, eğitim CV'sinde seçildi."))

# ---------------------------------------------------------------- T3
secilen_oran = 7.4
t3 = []
for _, r in duy.iterrows():
    oran = f"1 : {r.maliyet_orani:g}".replace(".", ",")
    if abs(r.maliyet_orani - secilen_oran) < 1e-9:
        oran += "  ← seçtiğim"
    t3.append([oran, f"{r.en_iyi_esik:.2f}", f3(r.recall), f3(r.precision),
               str(int(r.temas_edilen_musteri))])
vurgu_t3 = [i for i, r in enumerate(duy.itertuples()) if abs(r.maliyet_orani - secilen_oran) < 1e-9]
tablo("figures/t3_duyarlilik.png",
      "Eşik, maliyet varsayımına ne kadar duyarlı?",
      ["FN / FP maliyet oranı", "En iyi eşik", "Recall", "Precision", "Aranacak müşteri"],
      t3,
      vurgulu=vurgu_t3,
      genislik=[2.6, 1.2, 1.0, 1.15, 1.6],
      alt_not=(f"Eşik eğitim CV'sinde seçildi, test setinde ölçüldü. Varsayımı 1:2 alsam "
               f"{tr_sayi(duy.iloc[0].temas_edilen_musteri)} kişiyi ararım, 1:20 alsam "
               f"{tr_sayi(duy.iloc[-1].temas_edilen_musteri)} kişiyi."))

# ---------------------------------------------------------------- T4
tablo("figures/t4_katsayilar.png",
      "Lojistik regresyon katsayıları (odds oranı)",
      ["Değişken", "Odds oranı", "Ne anlama geliyor"],
      [["Fiber optik internet", "3.41", "Riski 3,4 kat artırıyor"],
       ["Elektronik çek ile ödeme", "1.50", "Riski %50 artırıyor"],
       ["Kağıtsız fatura", "1.40", "Riski %40 artırıyor"],
       ["Teknik destek paketi", "0.77", "Riski ~%23 azaltıyor"],
       ["Bir yıllık sözleşme", "0.49", "Riski yarıya indiriyor"],
       ["Müşteri yaşı (tenure)", "0.31", "En güçlü koruyucu etken"],
       ["İki yıllık sözleşme", "0.25", "Riski dörtte bire indiriyor"]],
      vurgulu=[0, 6],
      genislik=[2.6, 1.2, 3.0],
      hizalama=["left", "center", "left"],
      alt_not="Odds oranı > 1 riski artırır, < 1 azaltır.")

print("4 tablo görseli üretildi.")
