"""Adım 6 — Medium tablo desteklemediği için tabloları görsele çevir."""
import os as _os, pathlib as _p
KOK = _p.Path(__file__).resolve().parent.parent
_os.chdir(KOK)
for _d in ("outputs", "figures"):
    (KOK / _d).mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# ---------------------------------------------------------------- T1
tablo("figures/t1_cv.png",
      "5 katlı çapraz doğrulama sonuçları (eğitim seti)",
      ["Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "Accuracy"],
      [["Kukla (hep “ayrılmaz”)", "0.500", "0.265", "0.000", "0.000", "0.000", "0.735"],
       ["Lojistik Regresyon", "0.846", "0.662", "0.543", "0.654", "0.593", "0.802"],
       ["Lojistik Regresyon (dengeli)", "0.846", "0.660", "0.802", "0.519", "0.630", "0.750"],
       ["Random Forest", "0.826", "0.624", "0.484", "0.639", "0.550", "0.790"],
       ["Random Forest (dengeli)", "0.845", "0.658", "0.726", "0.563", "0.634", "0.778"]],
      vurgulu=[2],
      genislik=[3.3, 1.0, 1.0, 1.0, 1.05, 0.85, 1.05],
      alt_not="Kukla model %73,5 doğrulukla çalışıyor — ve hiçbir şey öğrenmiyor.")

# ---------------------------------------------------------------- T2
tablo("figures/t2_test.png",
      "Test seti sonuçları (1.409 müşteri)",
      ["Model", "ROC-AUC", "PR-AUC", "Recall (0.50)", "Precision (0.50)"],
      [["Lojistik Regresyon (dengeli)", "0.8417", "0.6331", "0.783", "0.505"],
       ["Random Forest (ayarlı)", "0.8455", "0.6580", "0.781", "0.529"]],
      vurgulu=[1],
      genislik=[3.3, 1.15, 1.1, 1.4, 1.5],
      alt_not="Aradaki fark 0,004 — çapraz doğrulama standart sapmasının yarısından az.")

# ---------------------------------------------------------------- T3
tablo("figures/t3_duyarlilik.png",
      "Eşik, maliyet varsayımına ne kadar duyarlı?",
      ["FN / FP maliyet oranı", "En iyi eşik", "Recall", "Precision", "Aranacak müşteri"],
      [["1 : 2", "0.53", "0.762", "0.544", "524"],
       ["1 : 3", "0.46", "0.824", "0.518", "595"],
       ["1 : 5", "0.34", "0.893", "0.448", "746"],
       ["1 : 7,4  ← seçtiğim", "0.23", "0.949", "0.394", "901"],
       ["1 : 10", "0.17", "0.979", "0.364", "1006"],
       ["1 : 20", "0.12", "0.992", "0.342", "1085"]],
      vurgulu=[3],
      genislik=[2.6, 1.2, 1.0, 1.15, 1.6],
      alt_not="Varsayımı 1:2 alsam 524 kişiyi ararım, 1:20 alsam 1.085 kişiyi.")

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
