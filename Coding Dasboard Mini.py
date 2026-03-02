import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

st.title("📊 Dashboard Analisis Data Siswa")

# ===============================
# LOAD DATA
# ===============================
file = "data_simulasi_50_siswa_20_soal.xlsx"
df = pd.read_excel(file)

soal_cols = [c for c in df.columns if "Soal" in c]

# ===============================
# DATA OTOMATIS
# ===============================
df["Total_Nilai"] = df[soal_cols].sum(axis=1)
df["Rata_siswa"] = df[soal_cols].mean(axis=1)
rata_soal = df[soal_cols].mean()

# ===============================
# SIDEBAR MENU
# ===============================
menu = st.sidebar.radio(
    "📂 Pilih Kategori Dashboard",
    [
        "Data Identitas",
        "Skor & Nilai Siswa",
        "Statistik Deskriptif",
        "Analisis Butir Soal",
        "Korelasi",
        "Regresi Linear",
        "Distribusi Data",
        "Diagram Lingkaran",
        "Grafik Analisis",
        "Kesimpulan"
    ]
)

# =====================================================
# 1. DATA IDENTITAS
# =====================================================
if menu == "Data Identitas":
    st.header("📋 Data Identitas (Kategorikal)")
    st.dataframe(df.select_dtypes(include="object"))

# =====================================================
# 2. SKOR SISWA
# =====================================================
elif menu == "Skor & Nilai Siswa":
    st.header("📊 Skor dan Nilai Siswa")
    st.dataframe(df[["Responden"] + soal_cols + ["Total_Nilai","Rata_siswa"]])

# =====================================================
# 3. STATISTIK DESKRIPTIF
# =====================================================
elif menu == "Statistik Deskriptif":
    st.header("📈 Statistik Deskriptif")

    statistik = df[soal_cols].describe().T
    statistik["Median"] = df[soal_cols].median()
    statistik["Modus"] = df[soal_cols].mode().iloc[0]

    st.dataframe(statistik)

# =====================================================
# 4. ANALISIS BUTIR SOAL
# =====================================================
elif menu == "Analisis Butir Soal":
    st.header("🧪 Rata-rata Tiap Soal")

    st.dataframe(rata_soal)

    fig, ax = plt.subplots()
    rata_soal.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rata-rata")
    st.pyplot(fig)

# =====================================================
# 5. KORELASI
# =====================================================
elif menu == "Korelasi":
    st.header("🔗 Heatmap Korelasi")

    corr = df[soal_cols].corr()

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# 6. REGRESI
# =====================================================
elif menu == "Regresi Linear":
    st.header("📉 Regresi Linear")

    X = df[["Rata_siswa"]]
    y = df["Total_Nilai"]

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot(X, pred)
    ax.set_xlabel("Rata-rata")
    ax.set_ylabel("Total Nilai")
    st.pyplot(fig)

    st.write("Koefisien:", model.coef_[0])
    st.write("Intercept:", model.intercept_)
    st.write("R²:", model.score(X, y))

# =====================================================
# 7. DISTRIBUSI DATA
# =====================================================
elif menu == "Distribusi Data":
    st.header("📊 Histogram Distribusi Nilai")

    fig, ax = plt.subplots()
    ax.hist(df["Total_Nilai"], bins=10)
    ax.set_xlabel("Total Nilai")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

# =====================================================
# 8. DIAGRAM LINGKARAN
# =====================================================
elif menu == "Diagram Lingkaran":
    st.header("🥧 Diagram Lingkaran Kategori Nilai")

    kategori = pd.cut(
        df["Total_Nilai"],
        bins=3,
        labels=["Rendah","Sedang","Tinggi"]
    )

    pie = kategori.value_counts()

    fig, ax = plt.subplots()
    ax.pie(pie, labels=pie.index, autopct="%1.1f%%")
    st.pyplot(fig)

# =====================================================
# 9. GRAFIK ANALISIS
# =====================================================
elif menu == "Grafik Analisis":
    st.header("📊 Grafik Analisis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Line Chart Skor Siswa")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["Total_Nilai"])
        st.pyplot(fig1)

    with col2:
        st.subheader("Bar Chart Rata-rata Soal")
        fig2, ax2 = plt.subplots()
        rata_soal.plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

# =====================================================
# 10. KESIMPULAN
# =====================================================
elif menu == "Kesimpulan":
    st.header("🧠 Kesimpulan Otomatis")

    mean_total = df["Total_Nilai"].mean()
    max_total = df["Total_Nilai"].max()

    if mean_total > 0.7 * max_total:
        st.success("Performa siswa kategori TINGGI")
    elif mean_total > 0.4 * max_total:
        st.warning("Performa siswa kategori SEDANG")
    else:
        st.error("Performa siswa kategori RENDAH")
