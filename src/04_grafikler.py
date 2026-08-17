"""Adım 4 — Yazı için grafikler."""
import os as _os, pathlib as _p
KOK = _p.Path(__file__).resolve().parent.parent
_os.chdir(KOK)
for _d in ("outputs", "figures"):
    (KOK / _d).mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# --- palet (dataviz referans paleti, açık mod) ---
YUZEY = "#fcfcfb"
MURK = "#0b0b0b"
IKINCIL = "#52514e"
SOLUK = "#898781"
IZGARA = "#e1e0d9"
EKSEN = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # mavi, turuncu, deniz yeşili
KRITIK = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": YUZEY, "axes.facecolor": YUZEY, "savefig.facecolor": YUZEY,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": EKSEN, "axes.linewidth": 1.0,
    "axes.labelcolor": IKINCIL, "text.color": MURK,
    "xtick.color": SOLUK, "ytick.color": SOLUK,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "grid.color": IZGARA, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def temizle(ax, y_izgara=True):
    ax.grid(axis="y" if y_izgara else "x", zorder=0)
    ax.set_axisbelow(True)


# ================================================================ veri
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
df["churn"] = (df["Churn"] == "Yes").astype(int)
y_te = np.load("outputs/y_test.npy")
p_lr = np.load("outputs/olasilik_lr.npy")
p_rf = np.load("outputs/olasilik_rf.npy")

# ================================================================ 1) sözleşme + tenure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

s = df.groupby("Contract")["churn"].mean().reindex(["Month-to-month", "One year", "Two year"]) * 100
etiket = ["Aydan aya", "1 yıllık", "2 yıllık"]
b = ax1.bar(etiket, s.values, color=[KRITIK, S2, S3], width=0.58, zorder=3)
for r, v in zip(b, s.values):
    ax1.text(r.get_x() + r.get_width() / 2, v + 1.4, f"%{v:.1f}",
             ha="center", fontsize=11, fontweight="bold", color=MURK)
ax1.set_ylim(0, 50)
ax1.set_ylabel("Ayrılma oranı")
ax1.yaxis.set_major_formatter(PercentFormatter())
ax1.set_title("Sözleşme tipi tek başına her şeyi anlatıyor",
              fontsize=12.5, fontweight="bold", color=MURK, pad=12, loc="left")
temizle(ax1)

df["kova"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72],
                    labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
t = df.groupby("kova", observed=True)["churn"].mean() * 100
ax2.plot(range(len(t)), t.values, color=S1, linewidth=2, marker="o",
         markersize=8, markerfacecolor=S1, markeredgecolor=YUZEY,
         markeredgewidth=2, zorder=3)
for i, v in enumerate(t.values):
    ax2.annotate(f"%{v:.0f}", (i, v), textcoords="offset points",
                 xytext=(0, 11), ha="center", fontsize=10.5,
                 fontweight="bold", color=MURK)
ax2.set_xticks(range(len(t)))
ax2.set_xticklabels(t.index)
ax2.set_xlabel("Müşteri yaşı (ay)")
ax2.set_ylim(0, 62)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.set_title("Risk ilk 6 ayda yoğunlaşıyor",
              fontsize=12.5, fontweight="bold", color=MURK, pad=12, loc="left")
temizle(ax2)
fig.savefig("figures/g1_segmentler.png")
plt.close(fig)

# ================================================================ 2) hizmet kırılımları
fig, ax = plt.subplots(figsize=(9, 5))
gruplar = [
    ("Fiber optik internet", df[df.InternetService == "Fiber optic"].churn.mean()),
    ("Elektronik çek ile ödeme", df[df.PaymentMethod == "Electronic check"].churn.mean()),
    ("Teknik destek paketi YOK", df[df.TechSupport == "No"].churn.mean()),
    ("Online güvenlik paketi YOK", df[df.OnlineSecurity == "No"].churn.mean()),
    ("65 yaş üstü", df[df.SeniorCitizen == 1].churn.mean()),
    ("TÜM MÜŞTERİLER", df.churn.mean()),
    ("Partneri var", df[df.Partner == "Yes"].churn.mean()),
    ("Bakmakla yükümlü olduğu kişi var", df[df.Dependents == "Yes"].churn.mean()),
    ("Otomatik kredi kartı ödemesi", df[df.PaymentMethod == "Credit card (automatic)"].churn.mean()),
]
gruplar = sorted(gruplar, key=lambda g: g[1])
ad = [g[0] for g in gruplar]
deg = [g[1] * 100 for g in gruplar]
ortalama = df.churn.mean() * 100
renk = [SOLUK if a == "TÜM MÜŞTERİLER" else (KRITIK if v > ortalama else S1)
        for a, v in zip(ad, deg)]
b = ax.barh(ad, deg, color=renk, height=0.62, zorder=3)
for r, v in zip(b, deg):
    ax.text(v + 0.9, r.get_y() + r.get_height() / 2, f"%{v:.1f}",
            va="center", fontsize=10.5, fontweight="bold", color=MURK)
ax.axvline(ortalama, color=EKSEN, linewidth=1.4, linestyle="--", zorder=2)
ax.text(ortalama + 0.7, 8.55, f"genel ortalama %{ortalama:.1f}",
        fontsize=9.5, color=IKINCIL, va="center")
ax.set_ylim(-0.7, 9.2)
ax.set_xlim(0, 52)
ax.xaxis.set_major_formatter(PercentFormatter())
ax.set_xlabel("Ayrılma oranı")
ax.set_title("Aynı veri setinde risk 3 kat değişiyor",
             fontsize=13, fontweight="bold", color=MURK, pad=14, loc="left")
ax.grid(axis="x", zorder=0)
ax.set_axisbelow(True)
fig.savefig("figures/g2_kirilimlar.png")
plt.close(fig)

# ================================================================ 3) ROC
fig, ax = plt.subplots(figsize=(6.4, 5.6))
for p, ad_, renk_ in [(p_lr, "Lojistik Regresyon", S1), (p_rf, "Random Forest", S2)]:
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, color=renk_, linewidth=2,
            label=f"{ad_} (AUC = {roc_auc_score(y_te, p):.3f})", zorder=3)
ax.plot([0, 1], [0, 1], color=EKSEN, linewidth=1.4, linestyle="--",
        label="Rastgele tahmin (AUC = 0.500)", zorder=2)
ax.set_xlabel("Yanlış alarm oranı (FPR)")
ax.set_ylabel("Yakalanan ayrılma oranı (TPR)")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.set_title("İki model neredeyse aynı yerde",
             fontsize=13, fontweight="bold", color=MURK, pad=14, loc="left")
ax.grid(zorder=0)
ax.set_axisbelow(True)
fig.savefig("figures/g3_roc.png")
plt.close(fig)

# ================================================================ 4) karmaşıklık matrisi
etiketler = [["Doğru: kalacak dedik", "Yanlış alarm"],
             ["KAÇAN müşteri", "Yakalanan"]]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
for ax, esik in zip(axes, [0.50, 0.23]):
    cm = confusion_matrix(y_te, (p_rf >= esik).astype(int))
    ax.pcolormesh(np.arange(3), np.arange(3), cm, cmap="Blues",
                  vmin=0, vmax=cm.max() * 1.3, edgecolors=YUZEY, linewidth=3)
    for i in range(2):
        for j in range(2):
            koyu = cm[i, j] > cm.max() * 0.6
            ax.text(j + 0.5, i + 0.62, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=19, fontweight="bold",
                    color="white" if koyu else MURK)
            ax.text(j + 0.5, i + 0.28, etiketler[i][j], ha="center", va="center",
                    fontsize=8.8, color="white" if koyu else IKINCIL)
    ax.set_xticks([0.5, 1.5], ["“kalır”", "“ayrılır”"])
    ax.set_yticks([0.5, 1.5], ["Kaldı", "Ayrıldı"])
    ax.set_xlabel("Modelin tahmini")
    ax.set_ylabel("Gerçekte ne oldu")
    ax.invert_yaxis()
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"Eşik {esik:.2f}  ·  {fn} müşteri kaçtı",
                 fontsize=12, fontweight="bold", color=MURK, pad=12, loc="left")
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
fig.subplots_adjust(wspace=0.28)
fig.savefig("figures/g4_matris.png")
plt.close(fig)

# ================================================================ 5) değişken önemi
pi = pd.read_csv("outputs/permutasyon_onem.csv").head(8).iloc[::-1]
tr = {"tenure": "Müşteri yaşı (ay)", "Contract": "Sözleşme tipi",
      "InternetService": "İnternet servisi", "TotalCharges": "Toplam harcama",
      "PaymentMethod": "Ödeme yöntemi", "MonthlyCharges": "Aylık ücret",
      "OnlineSecurity": "Online güvenlik", "TechSupport": "Teknik destek"}
fig, ax = plt.subplots(figsize=(8.4, 4.6))
b = ax.barh([tr.get(d, d) for d in pi.degisken], pi.onem, xerr=pi["std"],
            color=S1, height=0.6, zorder=3,
            error_kw=dict(ecolor=SOLUK, capsize=3, lw=1.2))
for r, v, s_ in zip(b, pi.onem, pi["std"]):
    ax.text(v + s_ + 0.0016, r.get_y() + r.get_height() / 2, f"{v:.3f}",
            va="center", fontsize=10, fontweight="bold", color=MURK)
ax.set_xlabel("ROC-AUC'de yaratılan düşüş (permütasyon önemi)")
ax.set_xlim(0, 0.05)
ax.set_title("Model dört değişkene yaslanıyor",
             fontsize=13, fontweight="bold", color=MURK, pad=14, loc="left")
ax.grid(axis="x", zorder=0)
ax.set_axisbelow(True)
fig.savefig("figures/g5_onem.png")
plt.close(fig)

# ================================================================ 6) desil / kümülatif yakalama
d = pd.DataFrame({"y": y_te, "p": p_rf})
d["desil"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10,
                     labels=[f"D{i}" for i in range(1, 11)])
t = d.groupby("desil", observed=True).agg(n=("y", "size"), c=("y", "sum")).reset_index()
t["oran"] = t.c / t.n * 100
t["kum"] = t.c.cumsum() / y_te.sum() * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
renk = [KRITIK if v >= 40 else (S1 if v >= 15 else SOLUK) for v in t.oran]
b = ax1.bar(range(1, 11), t.oran, color=renk, width=0.66, zorder=3)
for r, v in zip(b, t.oran):
    ax1.text(r.get_x() + r.get_width() / 2, v + 1.6, f"%{v:.0f}",
             ha="center", fontsize=9.5, fontweight="bold", color=MURK)
ax1.axhline(y_te.mean() * 100, color=EKSEN, linewidth=1.4, linestyle="--", zorder=2)
ax1.text(10.4, y_te.mean() * 100 + 1.5, f"ortalama\n%{y_te.mean()*100:.0f}",
         fontsize=9, color=IKINCIL, ha="right")
ax1.set_xticks(range(1, 11))
ax1.set_xticklabels([f"D{i}" for i in range(1, 11)])
ax1.set_xlabel("Risk skoruna göre dilim (D1 = en riskli %10)")
ax1.set_ylabel("Gerçekleşen ayrılma oranı")
ax1.set_ylim(0, 88)
ax1.yaxis.set_major_formatter(PercentFormatter())
ax1.set_title("En riskli dilimin dörtte üçü gerçekten gitti",
              fontsize=12, fontweight="bold", color=MURK, pad=12, loc="left")
temizle(ax1)

x = np.arange(0, 11) * 10
ax2.plot(x, np.r_[0, t.kum.values], color=S1, linewidth=2, marker="o",
         markersize=6, markerfacecolor=S1, markeredgecolor=YUZEY,
         markeredgewidth=1.8, label="Model sıralaması", zorder=3)
ax2.plot([0, 100], [0, 100], color=EKSEN, linewidth=1.4, linestyle="--",
         label="Rastgele arama", zorder=2)
ax2.annotate(f"%20'ye bakınca\nayrılanların %{t.kum.iloc[1]:.0f}'si",
             xy=(20, t.kum.iloc[1]), xytext=(33, 33),
             fontsize=10, color=MURK, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=IKINCIL, lw=1.2))
ax2.set_xlabel("Aranan müşteri yüzdesi")
ax2.set_ylabel("Yakalanan ayrılma yüzdesi")
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 102)
ax2.xaxis.set_major_formatter(PercentFormatter())
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.legend(loc="lower right", frameon=False, fontsize=10)
ax2.set_title("Bütçe sınırlıysa nereye bakmalı",
              fontsize=12, fontweight="bold", color=MURK, pad=12, loc="left")
ax2.grid(zorder=0)
ax2.set_axisbelow(True)
fig.savefig("figures/g6_desil.png")
plt.close(fig)

# ================================================================ 7) eşik-maliyet
KAYIP, TEKLIF = 893.0, 120.0
esikler = np.linspace(0.05, 0.95, 91)
maliyet, recall, precision = [], [], []
for e in esikler:
    tn, fp, fn, tp = confusion_matrix(y_te, (p_rf >= e).astype(int)).ravel()
    maliyet.append(fn * KAYIP + fp * TEKLIF)
    recall.append(tp / (tp + fn))
    precision.append(tp / (tp + fp) if (tp + fp) else np.nan)
maliyet = np.array(maliyet)
en_iyi = esikler[maliyet.argmin()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ax1.plot(esikler, maliyet / 1000, color=S1, linewidth=2, zorder=3)
ax1.scatter([en_iyi], [maliyet.min() / 1000], s=90, color=KRITIK,
            edgecolor=YUZEY, linewidth=2, zorder=4)
ax1.annotate(f"en iyi eşik {en_iyi:.2f}  ·  {maliyet.min()/1000:.0f}B $",
             xy=(en_iyi, maliyet.min() / 1000), xytext=(0.30, 158),
             fontsize=10.5, fontweight="bold", color=MURK,
             arrowprops=dict(arrowstyle="->", color=IKINCIL, lw=1.2))
i50 = np.argmin(np.abs(esikler - 0.50))
ax1.scatter([0.50], [maliyet[i50] / 1000], s=70, color=SOLUK,
            edgecolor=YUZEY, linewidth=2, zorder=4)
ax1.annotate(f"varsayılan 0.50  ·  {maliyet[i50]/1000:.0f}B $",
             xy=(0.50, maliyet[i50] / 1000), xytext=(0.30, 258),
             fontsize=10, color=IKINCIL,
             arrowprops=dict(arrowstyle="->", color=EKSEN, lw=1.1))
ax1.set_ylim(60, 355)
ax1.set_xlabel("Karar eşiği")
ax1.set_ylabel("Toplam maliyet (bin $)")
ax1.set_title("Varsayılan 0.50 en ucuz nokta değil",
              fontsize=12, fontweight="bold", color=MURK, pad=12, loc="left")
ax1.grid(zorder=0)
ax1.set_axisbelow(True)

ax2.plot(esikler, np.array(recall) * 100, color=S1, linewidth=2,
         label="Recall — ayrılanların kaçını yakaladık", zorder=3)
ax2.plot(esikler, np.array(precision) * 100, color=S2, linewidth=2,
         label="Precision — aradıklarımızın kaçı gerçekten gidecekti", zorder=3)
ax2.axvline(en_iyi, color=EKSEN, linewidth=1.4, linestyle="--", zorder=2)
ax2.text(en_iyi + 0.025, 66, f"seçilen eşik {en_iyi:.2f}", fontsize=9.5, color=IKINCIL)
ax2.set_xlabel("Karar eşiği")
ax2.set_ylabel("Oran")
ax2.set_ylim(0, 108)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.legend(loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(-0.02, -0.03))
ax2.set_title("Her kazanç bir takasla geliyor",
              fontsize=12, fontweight="bold", color=MURK, pad=12, loc="left")
ax2.grid(zorder=0)
ax2.set_axisbelow(True)
fig.savefig("figures/g7_esik.png")
plt.close(fig)

print("7 grafik üretildi.")
