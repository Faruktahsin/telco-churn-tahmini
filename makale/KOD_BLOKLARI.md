# Medium'daki 8 kod bloğu — sırayla

Yapıştırma sırasında bunlar düz paragraf olarak gelir. Her biri için Medium'da
boş bir satıra <code>```</code> yazıp **Enter**'a bas (gri kutu açılır), sonra
aşağıdaki metni içine yapıştır.

Sıra, yazıdaki geliş sırasıdır.

---

### 1 — "İlk kontrol: sınıf dengesi" bölümünde

```
df["Churn"].value_counts(normalize=True)
# No     0.7346
# Yes    0.2654
```

### 2 — Hemen ardından, TotalCharges boşlukları

```
bos = df["TotalCharges"].astype(str).str.strip() == ""
print(bos.sum())   # 11
print(df.loc[bos, "tenure"].unique())   # [0]
```

### 3 — "Ve en sevdiğim bulgu — cinsiyet" bölümünde

```
gender
Female  26.9%
Male    26.2%
```

### 4 — "Veri hazırlığı" bölümü, 2. madde

```
sayisal = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
kategorik = [c for c in X.columns if c not in sayisal]

on_isleme = ColumnTransformer([
    ("say", StandardScaler(), sayisal),
    ("kat", OneHotEncoder(drop="first", handle_unknown="ignore"), kategorik),
])
```

### 5 — "Bir korelasyon uyarısını not ettim" maddesi

```
                tenure  MonthlyCharges  TotalCharges
tenure           1.000           0.248         0.826
MonthlyCharges   0.248           1.000         0.651
TotalCharges     0.826           0.651         1.000
```

### 6 — "Doğruluk tuzağı" bölümü

```
Kukla (hep 'ayrılmaz')   accuracy = 0.7346   recall = 0.0   ROC-AUC = 0.500
```

### 7 — "Hiperparametre araması" bölümü

```
grid = GridSearchCV(
    pipeline,
    {"mdl__n_estimators": [300, 600],
     "mdl__max_depth": [6, 10, None],
     "mdl__min_samples_leaf": [1, 5, 15]},
    scoring="roc_auc", cv=cv, n_jobs=-1)
```

### 8 — "Beni yanıltan katsayı" bölümü

```
say__MonthlyCharges    katsayi = -0.555    odds_orani = 0.574
```

---

**Not:** 3, 5, 6 ve 8 numaralı bloklar kod değil, terminal çıktısı. Onları da gri
kutuya koymak doğru — Medium'da hizalı metin ancak böyle korunur, aksi halde
sütunlar kayar.
