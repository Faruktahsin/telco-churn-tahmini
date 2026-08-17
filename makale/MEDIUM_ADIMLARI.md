# Yazıyı Medium'a koyma adımları

## Önce bilmen gereken tek kısıt

**Medium tablo desteklemiyor.** Yazıdaki 4 tabloyu bu yüzden görsele çevirdim
(`t1`–`t4`). Medium'a görsel olarak yükleyeceksin, kimse farkı anlamaz —
aksine daha iyi görünür.

---

## Adım adım

**1. medium.com → sağ üstten profil → "Write a story"**

**2. `medium_kopyala.html` dosyasını tarayıcıda aç** (üzerine çift tıkla).
Medium'da nasıl görüneceğine yakın bir önizleme göreceksin.

**3. Sayfadaki her şeyi seç (Ctrl/Cmd + A), kopyala (Ctrl/Cmd + C),
Medium editörüne yapıştır (Ctrl/Cmd + V).**
Başlıklar, kalın yazılar, listeler ve ayraçlar korunur.

**4. En üstteki sarı uyarı kutusunu sil.** (Sadece senin için, yazıya ait değil.)

**5. Başlığı ve alt başlığı düzelt.**
Medium ilk satırı otomatik başlık yapar. Alt başlık satırını seçip çıkan
araç çubuğundan küçük **T** işaretine tıkla — böylece gri alt başlık stiline geçer.

**6. Görselleri kontrol et.**
Yapıştırma sırasında görseller genelde geliyor. Gelmediyse:
her görselin olması gereken yere tıkla, `+` işaretine bas, resim simgesini seç
ve aşağıdaki sıraya göre dosyayı yükle.

| Sıra | Dosya | Nerede |
|---|---|---|
| 1 | `g1_segmentler.png` | "Sonra kırılımlara baktım" bölümü |
| 2 | `g2_kirilimlar.png` | Aynı bölümün devamı |
| 3 | `t1_cv.png` | "Modeller ve çapraz doğrulama" |
| 4 | `g3_roc.png` | "Test seti: gerçek an" |
| 5 | `t2_test.png` | Hemen ROC grafiğinin altında |
| 6 | `g7_esik.png` | "0.50 nereden geliyor?" |
| 7 | `g4_matris.png` | Aynı bölümün devamı |
| 8 | `t3_duyarlilik.png` | "Bu varsayımlar ne kadar sağlam?" |
| 9 | `g6_desil.png` | "Peki bütçe sınırlıysa?" |
| 10 | `g5_onem.png` | "Model ne öğrendi" |
| 11 | `t4_katsayilar.png` | Aynı bölümün devamı |

**7. Kod bloklarını düzelt.**
Bu en çok uğraştıracak kısım. Medium'da kod bloğu oluşturmak için:
boş bir satıra <code>```</code> yaz ve **Enter**'a bas — gri kutu açılır,
kodu içine yapıştır. Yazıda 3 kod bloğu var (`value_counts`, `ColumnTransformer`,
`GridSearchCV`). Bunlar gri kutu olarak gelmediyse elle yeniden oluştur.

**8. Görsellere alt yazı ekle** (isteğe bağlı ama iyi durur).
Görsele tıklayınca altında "Type caption for image" çıkar.

**9. Sağ üstten "Publish" → 4-5 etiket ekle.**
Öneri: `Machine Learning`, `Data Science`, `Python`, `Veri Bilimi`, `Churn Prediction`

**10. Yayımladıktan sonra linki kopyala.**

---

## Sonra e-postayı gönder

**Kime:** info@turkiyeyapayzekaakademisi.com
**Konu:** `HSD Medium`
**İçerik:** Medium yazısının linki + siteye kayıtlı e-posta adresin

Ödev şartnamesi kodları göndermeni istemiyor — sadece link ve e-posta yeterli.

---

## Yayımlamadan önce son kontrol

- [ ] Sarı uyarı kutusu silindi mi?
- [ ] Alt başlık gri stile alındı mı?
- [ ] 11 görselin hepsi yerinde mi?
- [ ] 3 kod bloğu gri kutu içinde mi?
- [ ] En alttaki GitHub satırı: repo açtıysan linki koy, açmadıysan **satırı sil**
- [ ] Giriş ve kapanış kendi cümlelerinle mi?
