import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Applied AI Portfolio", page_icon="|", layout="wide")

st.markdown("""
    <style>
    /* ปรับแต่ง UI ให้ดูเคร่งขรึมและลึกลับขึ้น ลดความแวววาวลง */
    .main { 
        background-color: #020305; 
        color: #cfd8e3; 
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(8, 10, 15, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(75, 150, 200, 0.1);
        box-shadow: 10px 0 20px rgba(0, 0, 0, 0.7);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        background: transparent !important;
    }
    
    /* หัวข้อสีฟ้าไซเบอร์ ลบ Glow ที่ดูเป็นการ์ตูนออก */
    h1, h2, h3 { 
        color: #00ffff !important; 
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* ปรับแต่งปุ่มให้ดูแข็งแกร่งและดุดัน */
    .stButton>button { 
        width: 100%; 
        border-radius: 4px; /* ลดขอบโค้ง */
        height: 3.8em; 
        background-image: linear-gradient(135deg, #0a0e17 0%, #173b9e 70%, #00ffff 100%); 
        color: white; 
        border: 1px solid rgba(0, 255, 255, 0.2); 
        font-weight: 700; 
        font-size: 1.1em;
        transition: 0.2s ease-in-out;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton>button:hover { 
        transform: scale(1.01);
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.8);
        border: 1px solid #00ffff;
    }
    
    /* ปรับแต่งช่องกรอกข้อมูลให้ดูคมชัด */
    .stSelectbox, .stSlider, .stNumberInput { 
        background-color: #0a0e17; 
        border-radius: 4px; 
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: white;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #60a5fa !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ปรับแต่ง Alert และ Info Boxes ให้ดู Professional */
    .stAlert {
        border-radius: 6px;
        border: 1px solid;
    }
    
    .stAlert.st-b8 { /* Info */
        background-color: rgba(30, 64, 175, 0.2);
        border-color: #1e40af;
        color: #bfdbfe;
    }
    
    .stAlert.st-b9 { /* Success */
        background-color: rgba(16, 185, 129, 0.2);
        border-color: #10b981;
        color: #a7f3d0;
    }
    
    .stAlert.st-ba { /* Warning */
        background-color: rgba(245, 158, 11, 0.2);
        border-color: #f59e0b;
        color: #fef3c7;
    }
    
    .stAlert.st-bb { /* Error */
        background-color: rgba(239, 68, 68, 0.2);
        border-color: #ef4444;
        color: #fecaca;
    }
    
    .st-dg {
        background-color: #0a0e17;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    ml_model = pickle.load(open('ensemble_titanic.pkl', 'rb'))
    nn_model = load_model('nn_ai_vs_real.h5')
    sleep_model = pickle.load(open('sleep_model.pkl', 'rb'))
    return ml_model, nn_model, sleep_model

try:
    ml_model, nn_model, sleep_model = load_all_models()
except Exception as e:
    st.error(f"SYSTEM ERROR: ไม่พบไฟล์โมเดลที่จำเป็นบน GitHub. ตรวจสอบไฟล์ `.pkl` และ `.h5` ({e})")

st.sidebar.markdown("# | APPLIED AI HUB")
page = st.sidebar.radio("Navigation Bar", [
    "DASHBOARD | OVERVIEW",
    "ML Project | Titanic Theory", 
    "ML Project | Titanic Predictor",
    "NN Project | Image Classifier Theory",
    "NN Project | Image Classifier Predictor",
    "ANALYSIS | Sleep Health (Theory & Predict)"
])

if page == "DASHBOARD | OVERVIEW":
    st.title("Applied AI Project Hub")
    st.write("โครงการพัฒนาและทดสอบโมเดลปัญญาประดิษฐ์เพื่อการทำนายและวิเคราะห์ข้อมูลทางสถิติและรูปภาพ รายละเอียดและขั้นตอนการพัฒนาของแต่ละโมเดลอยู่ตามเส้นทางใน Navigation Bar ครับ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### Project ML: Titanic Survival\nพยากรณ์การรอดชีวิตจากเหตุการณ์เรือล่ม โดยอาศัยข้อมูลเชิงสถิติของผู้โดยสารจริง")
    with col2:
        st.info("### Project Analysis: Sleep Health Disorder\nวิเคราะห์ความเสี่ยงโรคที่เกิดจากการนอนหลับผ่านข้อมูลพฤติกรรมไลฟ์สไตล์")

elif page == "ML Project | Titanic Theory":
    st.title("Model Description: Titanic Survival Analysis")
    
    st.subheader("Data Source")
    st.write("Dataset ชุดนี้ได้รับอนุญาตให้ใช้จากเว็บไซต์ **Kaggle** (Titanic: Machine Learning from Disaster) ซึ่งเป็นฐานข้อมูลจริง")
    
    st.subheader("Features & Variables")
    st.markdown("""
    - **Pclass:** ชั้นของผู้โดยสาร (1st, 2nd, 3rd)
    - **Sex:** เพศของผู้โดยสาร (ปัจจัยหลักทางการสถิติ)
    - **Age:** อายุผู้โดยสาร
    - **Fare:** ราคาตั๋วเดินทาง
    """)
    
    st.subheader("Data Cleansing Pipeline")
    st.write("กระบวนการจัดการข้อมูลก่อนนำเข้าโมเดล:")
    st.markdown("""
    - **Imputation:** แทนที่ค่าว่างในช่อง 'Age' ด้วยค่า **Mean (ค่าเฉลี่ย)** ของข้อมูลทั้งหมด
    - **Label Encoding:** แปลงข้อมูล 'Male/Female' เป็นค่าตัวเลข **1 และ 0**
    - **Feature Selection:** ตัดข้อมูลที่ไม่มีผลต่อการทำนาย เช่น ชื่อ และเลขที่ตั๋วออกทั้งหมด
    """)
    
    st.subheader("Algorithm Structure")
    st.write("ใช้สถาปัตยกรรม **Ensemble Learning** โดยใช้ Voting Classifier ที่รวมเอาโมเดล Random Forest, XGBoost และ Logistic Regression เข้าไว้ด้วยกัน")

elif page == "ML Project | Titanic Predictor":
    st.title("Titanic Survival Predictor")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Pclass (1st, 2nd, 3rd)", [1, 2, 3])
        sex = st.radio("Gender", ["Male", "Female"])
        age = st.slider("Age", 0, 80, 25)
    with col2:
        fare = st.number_input("Fare (Ticket Price)", 0.0, 500.0, 32.0)
    
    if st.button("RUN PREDICTION"):
        sex_val = 1 if sex == "Male" else 0
        input_data = np.array([[pclass, sex_val, age, 0, 0, fare, 0]])
        res = ml_model.predict(input_data)
        if res[0] == 1:
            st.success("PREDICTION RESULT: SURVIVED")
        else:
            st.error("PREDICTION RESULT: NOT SURVIVED")

elif page == "NN Project | Image Classifier Theory":
    st.title("Model Description: AI vs Real Image Classifier")
    
    st.subheader("Data Source")
    st.write("ชุดรูปภาพทดสอบได้รับจากเว็บไซต์ **Kaggle** (rhythmghai/ai-vs-real-images-dataset)")
    
    st.subheader("Features")
    st.write("โมเดลวิเคราะห์จาก **Pixel-level Data** และลวดลายเชิงโครงสร้างของภาพเพื่อหาความแตกต่างระหว่างภาพถ่ายจริงกับภาพที่สร้างโดย AI")
    
    st.subheader("Image Preprocessing Pipeline")
    st.markdown("""
    - **Resizing:** ปรับขนาดรูปภาพทุกใบให้เป็น **128x128 พิกเซล** เพื่อความสม่ำเสมอของ Input Tensor
    - **Normalization:** ปรับค่าพิกเซลจาก 0-255 ให้เป็นช่วง **0-1** เพื่อเพิ่มเสถียรภาพในการเทรน Neural Network
    """)
    
    st.subheader("Algorithm Structure")
    st.write("ใช้โครงสร้าง **CNN (Convolutional Neural Network)** สำหรับการเรียนรู้ลวดลายเชิงลึกของภาพ")

elif page == "NN Project | Image Classifier Predictor":
    st.title("Image Classifier: AI vs Real")
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB').resize((128, 128))
        st.image(img, caption='Input Image', width=300)
        with st.spinner('Analyzing Image Structure...'):
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = nn_model.predict(img_array)
            score = prediction[0][0]
            label = "Prediction: AI Generated Image" if score < 0.5 else "Prediction: Real Image"
            st.subheader(f"{label}")

elif page == "ANALYSIS | Sleep Health (Theory & Predict)":
    st.title("Model Description: Sleep Disorder Analysis")
    
    with st.expander("Show Details (Theory & Cleansing Pipeline)"):
        st.subheader("Data Source")
        st.write("ข้อมูลมาจากเว็บไซต์ **Kaggle** (uom190346a/sleep-health-and-lifestyle-dataset)")
        
        st.subheader("Features")
        st.markdown("""
        - **Gender & Age**
        - **Sleep Duration**
        - **Physical Activity Level (0-100)**
        - **Stress Level (1-10)**
        - **BMI Category**
        """)
        
        st.subheader("Data Cleansing Pipeline")
        st.markdown("""
        - **Feature Mapping:** จัดกลุ่มและแปลงค่า BMI Category และอาชีพให้เป็นค่าตัวเลขกลุ่ม
        - **Managing Missing Data:** คอลัมน์ความผิดปกติทางการนอนที่ว่างอยู่ เราเติมค่า **'None'** เพื่อระบุว่าเป็นกลุ่มปกติ
        - **Normalization:** สำหรับ Features ที่เป็นตัวเลข เช่น ชั่วโมงนอน เพื่อให้โมเดลประมวลผลได้ดีขึ้น
        """)
        
        st.subheader("Algorithm Structure")
        st.write("ใช้สถาปัตยกรรม **Random Forest Classifier** สำหรับการเรียนรู้เงื่อนไขที่ซับซ้อน")

    st.divider()
    st.subheader("Sleep Health Analysis: Test Panel")
    col1, col2 = st.columns(2)
    with col1:
        s_gender = st.selectbox("Gender", ["Male", "Female"])
        s_age = st.number_input("Age", 10, 80, 30)
        s_dur = st.slider("Sleep Duration (Hours/Day)", 4.0, 10.0, 7.0)
    with col2:
        s_act = st.slider("Physical Activity Level (0-100)", 0, 100, 60)
        s_stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        s_bmi = st.selectbox("BMI Category", ["Normal", "Normal Weight", "Obese", "Overweight"])
    
    if st.button("RUN ANALYSIS"):
        bmi_map = {"Normal": 0, "Normal Weight": 1, "Obese": 2, "Overweight": 3}
        gen_val = 0 if s_gender == "Male" else 1
        s_input = np.array([[gen_val, s_age, s_dur, s_act, s_stress, bmi_map[s_bmi]]])
        
        try:
            s_res = sleep_model.predict(s_input)
            st.success(f"Analysis Result: {s_res[0]}")
            if s_res[0] != "None":
                st.warning("⚠️ WARNING: ข้อมูลบ่งชี้ถึงความเสี่ยง โปรดปรึกษาผู้เชี่ยวชาญทางการแพทย์เพื่อการวินิจฉัยที่ถูกต้อง")
        except Exception as e:
            st.error(f"Analysis Failed: {e}")
