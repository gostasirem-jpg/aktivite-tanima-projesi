import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Başlık ve Açıklama
st.title("İnsan Aktivitesi Tanıma Sistemi")
st.write("Sensör verilerini içeren test dosyanızı yükleyerek aktiviteleri tahmin edebilirsiniz.")

# Dosyaların varlığını güvenli bir şekilde kontrol etme
if not os.path.exists('en_iyi_model.pkl') or not os.path.exists('scaler_model.pkl'):
    st.error("Hata: 'en_iyi_model.pkl' veya 'scaler_model.pkl' dosyası GitHub deponuzda bulunamadı!")
    st.info("Lütfen bu dosyaları Colab'dan indirip GitHub deponuza yüklediğinizden emin olun.")
else:
    # Modelleri belleğe yükleme
    model = joblib.load('en_iyi_model.pkl')
    scaler = joblib.load('scaler_model.pkl')
    
    # LDA modelini kontrol etme (Varsa yükle, yoksa hata vermeden geç)
    lda = None
    if os.path.exists('lda_model.pkl'):
        lda = joblib.load('lda_model.pkl')

    # Kullanıcıdan CSV dosyası yüklemesini isteme
    yuklenen_dosya = st.file_uploader("Test Verisini Yükleyin (CSV)", type="csv")

    if yuklenen_dosya is not None:
        yeni_veri = pd.read_csv(yuklenen_dosya)
        
        st.write("Yüklenen Verinin İlk 5 Satırı:")
        st.dataframe(yeni_veri.head())
        
        # Olası metinsel sütunları (Activity gibi) ayıklamak için sadece sayısal sütunları seçiyoruz
        tahmin_verisi = yeni_veri.select_dtypes(include=[np.number])
        
        # Bilinen indeks ve kimlik sütunlarını temizleme
        for sutun in ['Activity', 'subject', 'Unnamed: 0']:
            if sutun in tahmin_verisi.columns:
                tahmin_verisi = tahmin_verisi.drop(columns=[sutun])
        
        # Modelin ve Scaler'ın tam olarak beklediği öznitelik sayısı (561)
        beklenen_boyut = scaler.n_features_in_
        
        # BOYUT UYUMSUZLUĞUNU OTOMATİK ÇÖZEN BLOK:
        # Eğer sütun sayısı hala beklenen boyuttan farklıysa, tam olarak ilk 561 sütunu kesip eşitleyelim
        if tahmin_verisi.shape[1] != beklenen_boyut:
            tahmin_verisi = tahmin_verisi.iloc[:, :beklenen_boyut]
        
        # Tahmin sürecini güvenli bir blokta çalıştırma
        try:
            # 1. Adım: Ölçeklendirme
            veri_scaled = scaler.transform(tahmin_verisi)
            
            # 2. Adım: Eğer LDA modeli yüklenmişse dönüşümü uygula, yoksa doğrudan devam et
            if lda is not None:
                veri_girdi = lda.transform(veri_scaled)
            else:
                veri_girdi = veri_scaled
                
            # 3. Adım: Tahmin Etme
            tahminler = model.predict(veri_girdi)
            
            # Sonuçları ana tabloya ekleme
            yeni_veri['Tahmin Edilen Aktivite'] = tahminler
            
            # Başarılı çıktıyı ekrana yazdırma
            st.success("Tahmin İşlemi Başarıyla Tamamlandı!")
            st.write("Aktivite Tahmin Sonuçları:")
            st.dataframe(yeni_veri[['Tahmin Edilen Aktivite']])
            
        except Exception as hata:
            st.error(f"Tahmin yapılırken teknik bir hata oluştu: {hata}")
            st.info(f"Sistem Bilgisi: Model {beklenen_boyut} özellik bekliyor. İşlenen veri boyutu: {tahmin_verisi.shape[1]}")
