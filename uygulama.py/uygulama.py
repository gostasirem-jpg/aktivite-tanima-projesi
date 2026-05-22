import streamlit as st
import pandas as pd
import joblib

# Başlık ve Açıklama
st.title("İnsan Aktivitesi Tanıma Sistemi")
st.write("Sensör verilerini içeren test dosyanızı yükleyerek aktiviteleri tahmin edebilirsiniz.")

# Kaydettiğimiz makine öğrenmesi modellerini belleğe alma
model = joblib.load('en_iyi_model.pkl')
scaler = joblib.load('scaler_model.pkl')
lda = joblib.load('lda_model.pkl')

# Kullanıcıdan CSV dosyası yüklemesini isteme
yuklenen_dosya = st.file_uploader("Test Verisini Yükleyin (CSV)", type="csv")

# Eğer kullanıcı bir dosya yüklediyse işlemleri sırasıyla başlat
if yuklenen_dosya is not None:
    # Veriyi okuma
    yeni_veri = pd.read_csv(yuklenen_dosya)
    
    st.write("Yüklenen Verinin İlk 5 Satırı:")
    st.dataframe(yeni_veri.head())
    
    # --- HATA ÇÖZÜM BLOĞU ---
    # Eğitilirken sildiğimiz sütunlar test verisinde varsa onları devre dışı bırakıyoruz
    tahmin_verisi = yeni_veri.copy()
    if 'Activity' in tahmin_verisi.columns:
        tahmin_verisi = tahmin_verisi.drop(['Activity'], axis=1)
    if 'subject' in tahmin_verisi.columns:
        tahmin_verisi = tahmin_verisi.drop(['subject'], axis=1)
    
    # Sırasıyla Eğitimde Yapılan İşlemleri Uygulama: Ölçeklendirme -> LDA -> Tahmin
    veri_scaled = scaler.transform(tahmin_verisi)
    veri_lda = lda.transform(veri_scaled)
    tahminler = model.predict(veri_lda)
    
    # Sonuçları orijinal tabloya ekleme
    yeni_veri['Tahmin Edilen Aktivite'] = tahminler
    
    # Ekrana yazdırma
    st.success("Tahmin İşlemi Başarıyla Tamamlandı!")
    st.write("Aktivite Tahmin Sonuçları:")
    st.dataframe(yeni_veri[['Tahmin Edilen Aktivite']])
