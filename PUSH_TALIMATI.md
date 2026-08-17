# GitHub'a yükleme — iki yol

Repo hazır: dosyalar düzenlendi, `git init` yapıldı, ilk commit atıldı.
Geriye sadece kendi GitHub hesabına bağlaman kaldı.

> Bu dosyayı push'tan sonra silebilirsin (repoya ait bir içerik değil).

---

## Yol A — Git kurulu ise (önerilen)

**1. GitHub'da boş repo aç**

github.com → sağ üstte **+** → **New repository**

- Repository name: `telco-churn-tahmini`
- Public
- ⚠️ **"Add a README file", "Add .gitignore", "Choose a license" kutularının
  ÜÇÜNÜ DE İŞARETSİZ BIRAK.** Repo tamamen boş olmalı, yoksa push çakışır.

**2. Terminalde klasöre gir ve push et**

```bash
cd telco-churn-tahmini

git remote add origin https://github.com/KULLANICI-ADIN/telco-churn-tahmini.git
git push -u origin main
```

`KULLANICI-ADIN` yerine kendi GitHub kullanıcı adını yaz.

**3. Şifre sorarsa**

GitHub artık hesap şifresini kabul etmiyor. Kullanıcı adını gir, şifre yerine
**Personal Access Token** yapıştır:

github.com → Settings → Developer settings → Personal access tokens →
Tokens (classic) → **Generate new token (classic)** → `repo` kutusunu işaretle →
Generate → çıkan token'ı kopyala.

Token sadece bir kez gösterilir, kaybedersen yenisini üretirsin.

### Commit'i kendi adına almak istersen

Commit şu an `Faruk Tahsin <afaruktahsin@gmail.com>` adına. Değiştirmek için
push'tan **önce**:

```bash
git config user.name "Adın Soyadın"
git config user.email "github-epostan@example.com"
git commit --amend --reset-author --no-edit
```

---

## Yol B — Git kurulu değilse (tarayıcıdan sürükle-bırak)

Git kurmak istemiyorsan tamamen tarayıcıdan da yapabilirsin.

1. github.com → **+** → **New repository** → adı `telco-churn-tahmini`, Public,
   **"Add a README file" işaretli** → Create
2. Açılan repoda **Add file → Upload files**
3. `telco-churn-tahmini` klasörünün **içindekileri** sürükleyip bırak
   (klasörün kendisini değil, içindeki `data`, `src`, `figures`, `outputs`,
   `makale` klasörlerini ve kök dosyaları)
4. GitHub'ın oluşturduğu boş `README.md`'nin üzerine yazmasına izin ver
5. Altta **Commit changes**

Bu yolda commit geçmişi olmaz ama repo aynı görünür.

---

## Push'tan sonra

**1. Medium yazısına repo linkini ekle.**
Yazının en altındaki `[repo linkinizi buraya ekleyin]` satırını gerçek linkle
değiştir:

```
Projenin kodları ve grafikleri GitHub'da:
https://github.com/KULLANICI-ADIN/telco-churn-tahmini
```

**2. README'ye Medium linkini ekle.**
`README.md`'nin başındaki `📄 **Medium yazısı:** [buraya linki ekleyin](#)`
satırını doldur. İkisi birbirine link verince daha derli toplu görünür.

**3. Repo açıklaması ve konu etiketleri ekle.**
Repo sayfasında sağ üstte ⚙️ → Description ve Topics:

> Telco müşteri kaybı tahmini — Lojistik Regresyon & Random Forest,
> maliyet tabanlı eşik optimizasyonu

Topics: `machine-learning` `churn-prediction` `scikit-learn` `python`
`data-science` `logistic-regression` `random-forest`

---

## Kontrol listesi

- [ ] Repo GitHub'da görünüyor
- [ ] README'deki 11 görsel yükleniyor (bozuk resim ikonu yok)
- [ ] Medium linki README'ye eklendi
- [ ] Repo linki Medium yazısına eklendi
- [ ] Bu dosya (`PUSH_TALIMATI.md`) silindi
