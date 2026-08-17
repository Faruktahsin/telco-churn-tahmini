# Telco Müşteri Kaybı (Churn) Tahmini

Bir telekom şirketinin 7.043 müşterisi üzerinde churn tahmini: keşifsel analiz,
Lojistik Regresyon ve Random Forest karşılaştırması, ve — projenin asıl bulgusu —
**karar eşiğinin model seçiminden çok daha önemli olduğunun** gösterilmesi.

Türkiye Yapay Zekâ Akademisi Veri Bilimi Bootcamp bitirme projesi.

📄 **Medium yazısı:** [buraya linki ekleyin](#)

---

## Özet bulgu

| | ROC-AUC | Test setinde kaçan müşteri |
|---|---|---|
| Lojistik Regresyon (dengeli) | 0.8417 | 81 |
| Random Forest (ayarlı) | **0.8455** | 82 |
| **Random Forest, eşik 0.50 → 0.23** | 0.8455 | **19** |

İki model arasındaki fark **0,004 AUC** — çapraz doğrulama standart sapmasının
(0,010) yarısından az, yani ölçüm gürültüsü. Aynı modelin karar eşiğini
varsayılan 0.50'den maliyet-optimal 0.23'e çekmek ise **63 müşteriyi kurtarıyor**
ve varsayılan maliyet modeline göre **21.939 $** tasarruf sağlıyor.

---

## Veri seti

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) —
7.043 müşteri, 21 değişken, %26,5 churn oranı.

Veri hazırlığında dikkat çeken nokta: `TotalCharges` sütununda 11 boş değer var
ve **hepsinin `tenure` değeri 0** — yani henüz ilk faturası kesilmemiş yeni
müşteriler. Rastgele bir eksiklik değil, verinin anlattığı bir durum; silmek
yerine 0 atandı.

---

## Bulgular

### Segment kırılımları

![Sözleşme tipi ve müşteri yaşına göre churn](figures/g1_segmentler.png)

Sözleşme tipi tek başına neredeyse her şeyi anlatıyor: aydan aya sözleşmelilerde
churn **%42,7**, iki yıllıklarda **%2,8**. Risk ilk 6 ayda yoğunlaşıyor (%52,9),
4 yılı devirenlerde %9,5'e iniyor.

![Gruplara göre churn oranı](figures/g2_kirilimlar.png)

Fiber optik kullananlarda %41,9, elektronik çekle ödeyenlerde %45,3.
Otomatik kredi kartı talimatı verenlerde ise sadece %15,2.

**Sinyal taşımayan değişken:** cinsiyet — kadınlarda %26,9, erkeklerde %26,2.
7.000 kişilik örneklemde gürültüden ibaret.

### Model karşılaştırması

![Çapraz doğrulama sonuçları](figures/t1_cv.png)

Taban çizgisi olarak konan kukla model (herkese "ayrılmaz" diyen) **%73,5
doğrulukla** çalışıyor — dengesiz veride accuracy metriğinin neden yanıltıcı
olduğunun somut kanıtı. Bu yüzden ana metrik ROC-AUC ve recall seçildi.

`class_weight="balanced"` ROC-AUC'yi hiç değiştirmiyor (0.8462 → 0.8460) ama
recall'ı 0.543'ten 0.802'ye çıkarıyor: sınıf ağırlıklandırması modelin sıralama
yeteneğini değil, kesme noktasını değiştiriyor.

![ROC eğrisi](figures/g3_roc.png)

### Karar eşiği — projenin asıl konusu

![Eşik ve maliyet](figures/g7_esik.png)

Basit bir maliyet modeliyle (kaçan müşteri ≈ 893 $, gereksiz teklif ≈ 120 $)
optimal eşik **0.23** çıkıyor.

![İki eşikte karmaşıklık matrisi](figures/g4_matris.png)

Eşik duyarlılığı da incelendi: maliyet oranı 1:2 alınırsa 524 müşteri, 1:20
alınırsa 1.085 müşteri aranıyor. Yani "kaç kişiyi arayalım" sorusunun cevabı
modelde değil, iş tarafının maliyet varsayımlarında.

![Eşiğin maliyet varsayımına duyarlılığı](figures/t3_duyarlilik.png)

### Operasyonel kullanım

![Desil analizi](figures/g6_desil.png)

En riskli %10'luk dilimde müşterilerin **%75,9'u** gerçekten ayrılmış (2,86 kat
lift). En riskli %20'ye bakarak ayrılacakların %50,5'i, %30'a bakarak %66,6'sı
yakalanıyor. Sınırlı bütçeli bir elde tutma ekibi için asıl kullanılabilir çıktı bu.

### Yorumlanabilirlik

![Permütasyon önemi](figures/g5_onem.png)
![Lojistik regresyon katsayıları](figures/t4_katsayilar.png)

İlginç bir ayrıntı: `MonthlyCharges` katsayısı **negatif** (odds oranı 0.574),
oysa ham veride churn edenlerin ortalama aylık ücreti daha yüksek (74,44 $ vs
61,27 $). Sebep çoklu doğrusal bağlantı — `InternetService_Fiber optic` değişkeni
"yüksek ücret + yüksek risk" ilişkisini zaten üstlenmiş durumda. Geriye kalan
etki, *aynı internet servisi içinde* daha çok ödeyen müşterinin daha bağlı
olduğunu ölçüyor.

---

## Çalıştırma

```bash
pip install -r requirements.txt

python src/01_kesif.py        # keşifsel analiz, kırılımlar, eksik değerler
python src/02_model.py        # modeller, çapraz doğrulama, eşik optimizasyonu
python src/03_duyarlilik.py   # maliyet duyarlılığı + desil (lift) analizi
python src/04_grafikler.py    # 7 grafik
python src/05_dogrulama.py    # yazıdaki 61 sayıyı bağımsız olarak yeniden hesaplar
python src/06_tablolar.py     # 4 tablo görseli
python src/07_html_uret.py    # yazının HTML sürümü
```

Scriptler nereden çağrılırsa çağrılsın kendi kökünü bulur; `cd` gerekmez.

### Doğrulama

`05_dogrulama.py`, Medium yazısındaki **61 sayısal iddiayı** ham veriden ve
kaydedilmiş model çıktılarından bağımsız olarak yeniden hesaplar. Çıktının son
satırı şu olmalı:

```
TOPLAM: 61 | HATA: 0
```

---

## Klasör yapısı

```
├── data/        Telco-Customer-Churn.csv
├── src/         7 analiz scripti
├── figures/     7 grafik + 4 tablo görseli
├── outputs/     model çıktıları (csv, json, npy)
└── makale/      Medium yazısı (md + html) ve yayın notları
```

---

## Sınırlar

Sonuçları okurken bilinmesi gerekenler:

1. **Veri kesitsel** — tarih bilgisi yok. Model "bu müşteri ne zaman gidecek"i
   değil, "ayrılmış müşterilere ne kadar benziyor"u cevaplıyor. Üretimde zamana
   göre bölme (geçmiş ayla eğit, sonraki ayla test et) gerekirdi.
2. **Korelasyon nedensellik değil** — iki yıllık sözleşmelilerin az ayrılması,
   uzun sözleşmenin sadakat *yarattığını* göstermez; zaten sadık müşteriler uzun
   sözleşme imzalıyor olabilir.
3. **Maliyet varsayımları tahminî** — 893 $ ve 120 $ rakamları bu çalışmaya ait
   varsayımlar. Duyarlılık analizi bu yüzden eklendi.
4. **Tek bölme** — farklı bir `random_state` ile ROC-AUC ±0,01 oynayabilir.
5. **Kalibrasyon kontrol edilmedi** — Random Forest olasılıkları genelde iyi
   kalibre olmaz. Sıralama için sorun değil, ama olasılıkları doğrudan maliyet
   hesabına sokmak isteseydik `CalibratedClassifierCV` gerekirdi.

## Sonraki adımlar

- Gradient boosting (XGBoost / LightGBM) karşılaştırması
- SHAP ile müşteri bazında açıklama üretimi
- Olasılık kalibrasyonu
- Gerçek maliyet rakamlarının iş tarafından alınması

---

## Ortam

pandas 3.0.2 · scikit-learn 1.8.0 · matplotlib 3.10.9 · Python 3.11
Tüm scriptlerde `random_state=42` sabit.

## Lisans

MIT — bkz. [LICENSE](LICENSE). Veri seti IBM'e aittir, kendi lisansına tabidir.
