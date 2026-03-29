import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

# --- โหลดโมเดล (ตรวจสอบชื่อไฟล์ให้ตรงกับที่คุณโหลดจาก Colab) ---
@st.cache_resource
def load_my_models():
    ml_model = pickle.load(open('ensemble_titanic.pkl', 'rb'))
    nn_model = load_model('nn_ai_vs_real_v2.keras')
    return ml_model, nn_model

# ป้องกัน Error ถ้ายังไม่มีไฟล์โมเดล
try:
    ml_model, nn_model = load_my_models()
except:
    st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบชื่อไฟล์บน GitHub")

# --- Sidebar Menu ---
st.sidebar.title("🚢 AI Project Dashboard")
page = st.sidebar.radio("Go to", ["ML Theory", "ML Testing", "NN Theory", "NN Testing"])

# --- Page 1: ML Theory (Titanic) ---
if page == "ML Theory":
    st.title("Titanic Survival Analysis")
    st.subheader("1. Dataset Info")
    st.write("Source: Kaggle (Titanic Dataset)")
    st.write("Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")
    
    # โชว์ข้อมูลดิบ (Data Cleaning)
    df = pd.read_csv('Titanic-Dataset.csv')
    st.write("Sample Data:", df.head())
    
    st.subheader("2. Ensemble Model Structure")
    st.write("เราใช้ Voting Classifier ที่ประกอบด้วย 3 โมเดลย่อย: Random Forest, XGBoost, และ Logistic Regression")

# --- Page 2: ML Testing ---
elif page == "ML Testing":
    st.title("Test Titanic Prediction")
    pclass = st.selectbox("Class (1=First, 3=Third)", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", 0, 100, 25)
    fare = st.number_input("Fare", 0.0, 500.0, 32.0)
    
    if st.button("Predict"):
        # แปลงข้อมูลเป็นตัวเลขก่อนเข้าโมเดล
        sex_val = 1 if sex == "Male" else 0
        input_data = np.array([[pclass, sex_val, age, 0, 0, fare, 0]]) # ปรับให้ครบ 7 features
        res = ml_model.predict(input_data)
        st.success(f"Result: {'Survived' if res[0]==1 else 'Not Survived'}")

# --- Page 3: NN Theory (AI vs Real) ---
elif page == "NN Theory":
    st.title("AI vs Real Image Classification")
    st.write("Source: Kaggle (rhythmghai/ai-vs-real-images-dataset)")
    st.write("Algorithm: CNN (Convolutional Neural Network) with Transfer Learning (MobileNetV2)")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png", caption="CNN Architecture Example")

# --- Page 4: NN Testing ---
elif page == "NN Testing":
    st.title("Test Image Classification")
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).resize((128, 128))
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # เตรียมภาพก่อน Predict
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = nn_model.predict(img_array)
        label = "AI Generated" if prediction[0][0] < 0.5 else "Real Image"
        st.write(f"### Prediction: {label}")
        st.write(f"Confidence Score: {prediction[0][0]:.4f}")