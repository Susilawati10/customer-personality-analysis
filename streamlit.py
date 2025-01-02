import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("modelXGB.joblib")
scaler = joblib.load("scaler.joblib")

# Judul aplikasi
st.title("Customer Personality Analysis")

# Input data
education = st.selectbox(
    "Education",
    ("Basic", "2nd Cycle", "Graduation", "Master", "PhD")
)
education_map = {"Basic": 1, "2nd Cycle": 2, "Graduation": 3, "Master": 4, "PhD": 5}
education_numeric = education_map[education]

marital_status = st.selectbox(
    "Marital Status",
    ("Single", "Married", "Divorced")
)
marital_status_map = {"Single": 1, "Married": 2, "Divorced": 3}
marital_status_numeric = marital_status_map[marital_status]

income = st.number_input("Income", min_value=1730.0, max_value=118454.75)
teenhome = st.number_input("Teenhome", min_value=0, max_value=2, step=1)
recency = st.number_input("Recency", min_value=0, max_value=99, step=1)
mntwines = st.number_input("MntWines", min_value=0.0, max_value=1226.50)
mntfruits = st.number_input("MntFruits", min_value=0.0, max_value=79.50)
mntmeatproducts = st.number_input("MntMeatProducts", min_value=0.0, max_value=556.625)
mntfishproducts = st.number_input("MntFishProducts", min_value=0.0, max_value=120.50)
mntsweetproducts = st.number_input("MntSweetProducts", min_value=0.0, max_value=81.0)
mntgoldprods = st.number_input("MntGoldProds", min_value=0.0, max_value=126.50)
numwebpurchases = st.number_input("NumWebPurchases", min_value=0, max_value=12, step=1)
numcatalogpurchases = st.number_input("NumCatalogPurchases", min_value=0, max_value=10, step=1)
numstorepurchases = st.number_input("NumStorePurchases", min_value=0, max_value=13, step=1)
total_mnt = st.number_input("Total Mnt", min_value=5.0, max_value=2516.50)
total_purchases = st.number_input("Total Purchases", min_value=0.0, max_value=40.50)
everacceptedcmp = st.selectbox("EverAcceptedCmp", (0, 1, 2, 3, 4, 5))
days_enrolled = st.number_input("Days Enrolled", min_value=0, max_value=699, step=1)

# Collect data input untuk prediksi
input_data = np.array([
    education_numeric,
    marital_status_numeric,
    income,
    teenhome,
    recency,
    mntwines,
    mntfruits,
    mntmeatproducts,
    mntfishproducts,
    mntsweetproducts,
    mntgoldprods,
    numwebpurchases,
    numcatalogpurchases,
    numstorepurchases,
    total_mnt,
    total_purchases,
    everacceptedcmp,
    days_enrolled
]).reshape(1, -1)

# Transform data dengan scaler
scaled_data = scaler.transform(input_data)

prediction = model.predict(scaled_data)

# Tombol prediksi
if st.button("Predict"):
    if prediction == 0:
        st.markdown('Customer reject campaign')
    elif prediction == 1:
        st.markdown('Customer accept campaign')