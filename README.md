# Bitirme Projesi (Graduation Project)

Bu proje son sınıf bitirme projesi için oluşturulmuş çalışma alanıdır.

## Proje Yapısı

```
bitirme/
│
├── .venv/                   # Python sanal ortamı (virtual environment)
├── train_images/            # Araç görsel verileri (sınıf bazında klasörlenmiş)
├── train_data.csv           # Eğitim etiketleri ve görsel dosya eşleşmeleri
├── train_features.csv       # Önceden çıkarılmış görsel öznitelikleri (features)
├── notebook.ipynb           # Proje başlangıç ve keşifçi veri analizi not defteri
├── requirements.txt         # Gerekli Python kütüphaneleri
├── .gitignore               # Git tarafından takip edilmeyecek dosya ve klasörler
└── README.md                # Proje açıklaması ve kullanım kılavuzu
```

---

## Kurulum ve Başlatma

### 1. Sanal Ortamı (venv) Aktif Etme

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

*(Eğer PowerShell script çalıştırma kısıtlaması uyarısı verirse `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` komutunu çalıştırabilirsiniz).*

Windows Komut İstemi (CMD):
```cmd
.\.venv\Scripts\activate.bat
```

### 2. Kütüphaneleri Yükleme

Gerekli ek kütüphaneleri yüklemek için:
```bash
pip install -r requirements.txt
```

### 3. Jupyter Notebook Başlatma

Tarayıcı üzerinden Jupyter Notebook arayüzünü açmak için:
```bash
jupyter notebook
```
veya doğrudan IDE içinde `notebook.ipynb` dosyasını açıp sağ üst köşeden `.venv (Python 3.14)` çekirdeğini (kernel) seçebilirsiniz.

---

## Git Kullanımı

Yapılan değişiklikleri kaydetmek için:
```bash
# Değişiklikleri inceleme
git status

# Dosyaları sahneye ekleme
git add .

# Yeni bir commit oluşturma
git commit -m "feat: proje baslangic commit'i"

# Uzak depoya bağlama (GitHub / GitLab repo açtıktan sonra):
# git remote add origin <REPO_URL>
# git branch -M main
# git push -u origin main
```
