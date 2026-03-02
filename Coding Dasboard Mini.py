import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

st.title("📊 DASHBOARD ANALISIS DATA SISWA")

# ===============================
# LOAD DATA
# ===============================
file = "data_simulasi_50_siswa_20_soal.xlsx"
df = pd.read_excel(file)

st.subheader("📋 Data Identitas & Skor")
st.dataframe(df)

# ===============================
# IDENTIFIKASI DATA
# ===============================
soal_cols = [c for c in df.columns if "Soal" in c]

# ===============================
# DATA OTOMATIS
# ===============================
df["Total_Nilai"] = df[soal_cols].sum(axis=1)
df["Rata_siswa"] = df[soal_cols].mean(axis=1)

rata_soal = df[soal_cols].mean()

# ===============================
# STATISTIK DESKRIPTIF
# ===============================
st.header("📈 Statistik Deskriptif")

statistik = df[soal_cols].describe().T
statistik["median"] = df[soal_cols].median()
statistik["modus"] = df[soal_cols].mode().iloc[0]

st.dataframe(statistik)

# ===============================
# TABEL ANALISIS
# ===============================
st.header("📋 Tabel Analisis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tabel Skor Siswa")
    st.dataframe(df[["Responden","Total_Nilai","Rata_siswa"]])

with col2:
    st.subheader("Rata-rata Soal")
    st.dataframe(rata_soal)

# ===============================
# BAR CHART RATA-RATA SOAL
# ===============================
st.header("📊 Grafik Rata-rata Soal")

fig1, ax1 = plt.subplots()
rata_soal.plot(kind="bar", ax=ax1)
ax1.set_ylabel("Rata-rata")
st.pyplot(fig1)

# ===============================
# LINE CHART SKOR SISWA
# ===============================
st.header("📈 Grafik Skor Total Siswa")

fig2, ax2 = plt.subplots()
ax2.plot(df["Total_Nilai"])
ax2.set_xlabel("Siswa")
ax2.set_ylabel("Total Nilai")
st.pyplot(fig2)

# ===============================
# DISTRIBUSI DATA (HISTOGRAM)
# ===============================
st.header("📉 Distribusi Nilai")

fig3, ax3 = plt.subplots()
ax3.hist(df["Total_Nilai"], bins=10)
ax3.set_xlabel("Total Nilai")
ax3.set_ylabel("Frekuensi")
st.pyplot(fig3)

# ===============================
# DIAGRAM LINGKARAN
# ===============================
st.header("🥧 Diagram Lingkaran Kategori Nilai")

kategori = pd.cut(
    df["Total_Nilai"],
    bins=3,
    labels=["Rendah", "Sedang", "Tinggi"]
)

pie_data = kategori.value_counts()

fig4, ax4 = plt.subplots()
ax4.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
st.pyplot(fig4)

# ===============================
# KORELASI
# ===============================
st.header("🔗 Heatmap Korelasi")

corr = df[soal_cols].corr()

fig5, ax5 = plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax5)
st.pyplot(fig5)

# ===============================
# REGRESI LINEAR
# ===============================
st.header("📉 Regresi Linear")

X = df[["Rata_siswa"]]
y = df["Total_Nilai"]

model = LinearRegression()
model.fit(X, y)

prediksi = model.predict(X)

fig6, ax6 = plt.subplots()
ax6.scatter(X, y)
ax6.plot(X, prediksi)
ax6.set_xlabel("Rata-rata Siswa")
ax6.set_ylabel("Total Nilai")
st.pyplot(fig6)

st.write("Koefisien Regresi:", model.coef_[0])
st.write("Intercept:", model.intercept_)
st.write("R² Score:", model.score(X, y))

# ===============================
# KESIMPULAN OTOMATIS
# ===============================
st.header("🧠 Kesimpulan Otomatis")

mean_total = df["Total_Nilai"].mean()

if mean_total > df["Total_Nilai"].max()*0.7:
    kesimpulan = "Performa siswa secara umum TINGGI."
elif mean_total > df["Total_Nilai"].max()*0.4:
    kesimpulan = "Performa siswa berada pada kategori SEDANG."
else:
    kesimpulan = "Performa siswa masih RENDAH."

st.success(kesimpulan)
