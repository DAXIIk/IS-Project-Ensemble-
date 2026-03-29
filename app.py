import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Multi-Project Dashboard", page_icon="🤖", layout="wide")

# --- 1. ส่วนโหลดโมเดลทั้งหมด ---
@st.cache_resource
def load_all_models():
    # โหลด Titanic (ML)
    ml_model = pickle.load(open('ensemble_titanic.pkl', 'rb'))
    # โหลด AI vs Real (NN)
    nn_model = load_model('nn_ai_vs_real.h5')
    # โหลด Sleep Health (Model ตัวใหม่)
    sleep_model = pickle.load(open('sleep_model.pkl', 'rb'))
    return ml_model, nn_model, sleep_model

# จัดการ Error กรณีไฟล์หาย
try:
    ml_model, nn_model, sleep_model = load_all_models()
except Exception as e:
    st.error(f"⚠️ ไม่พบไฟล์โมเดลบางส่วนบน GitHub: {e}")

# --- 2. Sidebar Menu ---
st.sidebar.title("🚢 AI Project Dashboard")
page = st.sidebar.radio("เลือกหัวข้อ", [
    "ML Titanic: Theory", "ML Titanic: Testing", 
    "NN Image: Theory", "NN Image: Testing",
    "Sleep Health: AI Predictor"
])

# --- Page 1: ML Theory (Titanic) ---
if page == "ML Titanic: Theory":
    st.title("🚢 Titanic Survival Analysis")
    st.subheader("1. Data Cleansing & Preparation")
    
    # โชว์กระบวนการ Clean ข้อมูล
    st.markdown("""
    **ขั้นตอนการเตรียมข้อมูล (Data Cleansing):**
    * **Fill Missing Values:** เติมค่าว่างในช่อง 'Age' ด้วยค่าเฉลี่ย (Mean)
    * **Feature Encoding:** แปลงเพศ (Male/Female) ให้เป็นตัวเลข (1/0)
    * **Feature Selection:** เลือกเฉพาะปัจจัยที่ส่งผลต่อการรอดชีวิต (Pclass, Sex, Age, Fare)
    """)
    
    # โชว์ข้อมูลดิบ
    try:
        df = pd.read_csv('train.csv')
        st.write("ตัวอย่างข้อมูลหลัง Clean (บางส่วน):", df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].head())
    except:
        st.warning("ไม่พบไฟล์ train.csv ใน Repository")

# --- Page 2: ML Testing (Titanic) ---
elif page == "ML Titanic: Testing":
    st.title("🔮 Titanic Prediction Test")
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Class (1=ชั้นหนึ่ง, 3=ชั้นประหยัด)", [1, 2, 3])
        sex = st.radio("เพศ", ["Male", "Female"])
        age = st.slider("อายุ", 0, 100, 25)
        fare = st.number_input("ราคาตั๋ว (Fare)", 0.0, 500.0, 32.0)
    
    if st.button("ทำนายผลการรอดชีวิต"):
        sex_val = 1 if sex == "Male" else 0
        # ส่งค่าไป 7 features ตามที่โมเดลต้องการ [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
        input_data = np.array([[pclass, sex_val, age, 0, 0, fare, 0]]) 
        res = ml_model.predict(input_data)
        
        if res[0] == 1:
            st.success("ผลลัพธ์: ✅ Survived (รอดชีวิต)")
            st.balloons()
        else:
            st.error("ผลลัพธ์: ❌ Not Survived (เสียชีวิต)")

# --- Page 3: NN Theory ---
elif page == "NN Image: Theory":
    st.title("🖼️ AI vs Real Image Theory")
    st.write("**Algorithm:** Convolutional Neural Network (CNN)")
    st.write("**Preprocessing:**")
    st.markdown("""
    1. **Resizing:** ปรับขนาดรูปภาพเป็น 128x128 พิกเซล
    2. **Normalization:** หารค่าพิกเซลด้วย 255 เพื่อทำให้อยู่ในกลุ่ม 0-1
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png", caption="โครงสร้าง CNN")

# --- Page 4: NN Testing ---
elif page == "NN Image: Testing":
    st.title("📷 AI vs Real Classifier")
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพเพื่อตรวจสอบ...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB').resize((128, 128))
        st.image(img, caption='รูปที่อัปโหลด', width=300)
        
        # Preprocessing
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = nn_model.predict(img_array)
        score = prediction[0][0]
        label = "AI Generated" if score < 0.5 else "Real Image"
        
        st.subheader(f"ผลทำนาย: {label}")
        st.write(f"Confidence Score: {score:.4f}")

# --- Page 5: Sleep Health ---
elif page == "Sleep Health: AI Predictor":
    st.title("💤 Sleep Disorder Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        s_gender = st.selectbox("เพศ", ["Male", "Female"])
        s_age = st.number_input("อายุ (ปี)", 10, 100, 30)
        s_dur = st.slider("ชั่วโมงการนอน", 4.0, 10.0, 7.0)
    with col2:
        s_act = st.number_input("ระดับกิจกรรมทางกาย (Physical Activity Level)", 0, 100, 60)
        s_stress = st.slider("ระดับความเครียด (Stress Level 1-10)", 1, 10, 5)
        s_bmi = st.selectbox("กลุ่ม BMI", ["Normal", "Overweight", "Obese"])
    
    if st.button("วิเคราะห์สุขภาพการนอน"):
        # 1. Cleansing ข้อมูลให้ตรงกับที่เทรนใน Colab
        gender_val = 0 if s_gender == "Male" else 1 # ตาม map({'Male': 0, 'Female': 1})
        
        # BMI ใน Colab คุณใช้ .cat.codes ซึ่งมักจะเรียงตามตัวอักษร: Normal=0, Obese=1, Overweight=2
        # (แนะนำให้เช็คใน Colab อีกทีเพื่อความชัวร์ แต่เบื้องต้นลองตามนี้ครับ)
        # แก้ไข bmi_map ให้ตรงกับลำดับใน Colab (Index 0-3)
        bmi_map = {
            "Normal": 0, 
            "Normal Weight": 1, 
            "Obese": 2, 
            "Overweight": 3
        } 
        
        # 2. จัดเรียง Features ให้ครบ 6 ตัวตามที่โมเดลต้องการเป๊ะๆ
        # ลำดับ: Gender, Age, Sleep Duration, Physical Activity Level, Stress Level, BMI Category
        try:
            s_input = np.array([[
                gender_val, 
                s_age, 
                s_dur, 
                s_act, 
                s_stress, 
                bmi_map[s_bmi]
            ]])
            
            s_res = sleep_model.predict(s_input)
            
            st.success(f"### ผลการทำนาย: {s_res[0]}")
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
