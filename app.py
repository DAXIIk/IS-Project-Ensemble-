import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

# --- ตั้งค่าหน้าเว็บแบบเท่ๆ Dark Mode ---
st.set_page_config(page_title="AI Project Portfolio", page_icon="🌙", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { 
        width: 100%; border-radius: 10px; height: 3.5em; 
        background-image: linear-gradient(to right, #1e40af, #3b82f6); 
        color: white; border: none; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); opacity: 0.9; }
    h1, h2, h3 { color: #60a5fa; }
    .stSelectbox, .stSlider, .stNumberInput { background-color: #1f2937; border-radius: 8px; }
    .sidebar .sidebar-content { background-color: #111827; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. ส่วนโหลดโมเดล (Cache ไว้จะได้ลื่นๆ) ---
@st.cache_resource
def load_all_models():
    ml_model = pickle.load(open('ensemble_titanic.pkl', 'rb'))
    nn_model = load_model('nn_ai_vs_real.h5')
    sleep_model = pickle.load(open('sleep_model.pkl', 'rb'))
    return ml_model, nn_model, sleep_model

try:
    ml_model, nn_model, sleep_model = load_all_models()
except Exception as e:
    st.error(f"เฮ้ย! หาไฟล์โมเดลไม่เจอว่ะ เช็คชื่อไฟล์บน GitHub แป๊บนึงนะ: {e}")

# --- 2. Sidebar Menu ---
st.sidebar.title("🚀 AI Multi-Project")
page = st.sidebar.radio("เลือกหน้าที่จะไป:", [
    "🏠 Dashboard Overview",
    "📊 Titanic Theory (ML)", 
    "🔮 Titanic Testing",
    "🧠 AI vs Real Theory (NN)",
    "📷 AI vs Real Testing",
    "💤 Sleep Health Theory & Predict"
])

# --- PAGE: DASHBOARD OVERVIEW ---
if page == "🏠 Dashboard Overview":
    st.title("🌙 ยินดีต้อนรับเข้าสู่โปรเจค AI!")
    st.write("เว็บนี้รวมโมเดลที่เราตั้งใจทำ ทั้งสายสถิติและสายภาพ ลองเลือกเล่นที่แถบซ้ายมือได้เลยครับ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📂 ชุดข้อมูลที่ 1: Titanic\nเน้นพยากรณ์การรอดชีวิตจากเหตุการณ์เรือล่ม โดยใช้สถิติผู้โดยสารจริง")
    with col2:
        st.info("### 📂 ชุดข้อมูลที่ 2: Sleep Health\nวิเคราะห์พฤติกรรมการนอน เพื่อดูความเสี่ยงโรคต่างๆ เช่น นอนไม่หลับ")

# --- PAGE: ML TITANIC THEORY ---
elif page == "📊 Titanic Theory (ML)":
    st.title("📊 เจาะลึกทฤษฎีไททานิค")
    st.subheader("📍 ที่มาของข้อมูล")
    st.write("โหลดมาจาก Kaggle ครับ เป็น Dataset ระดับตำนานที่คนหัดทำ AI ต้องผ่านทุกคน!")
    
    st.subheader("🧹 การเตรียมข้อมูล (Data Cleansing)")
    st.write("ข้อมูลมันไม่ได้มาสวยๆ นะครับ เราต้องจัดการก่อน:")
    st.markdown("""
    * **จัดการค่าว่าง:** ใครไม่ยอมบอกอายุ เราเอา 'อายุเฉลี่ย' ของทุกคนไปเติมให้
    * **แปลงร่างข้อมูล:** AI อ่านคำว่า 'ชาย/หญิง' ไม่เป็น เราเลยเปลี่ยนเป็นเลข 1 กับ 0
    * **คัดเลือกปัจจัย:** เราหยิบแค่ Pclass, Sex, Age, Fare มาใช้ เพราะพวกนี้แหละตัวตัดสินชีวิต
    """)
    
    st.subheader("🧠 อัลกอริทึม: Ensemble Learning")
    st.write("เราใช้ท่า **Voting Classifier** ครับ คือเอาโมเดล 3 ตัว (Random Forest, XGBoost, Logistic) มาโหวตกันว่าใครจะรอด")

# --- PAGE: ML TITANIC TESTING ---
elif page == "🔮 Titanic Testing":
    st.title("🔮 มาลองทำนายกันว่าจะรอดไหม?")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("ชั้นที่นั่ง (1=VIP, 3=ประหยัด)", [1, 2, 3])
        sex = st.radio("เพศ", ["Male", "Female"])
        age = st.slider("อายุเท่าไหร่?", 0, 80, 25)
    with col2:
        fare = st.number_input("ราคาตั๋ว (Fare)", 0.0, 500.0, 32.0)
    
    if st.button("PREDICT NOW"):
        sex_val = 1 if sex == "Male" else 0
        input_data = np.array([[pclass, sex_val, age, 0, 0, fare, 0]])
        res = ml_model.predict(input_data)
        if res[0] == 1:
            st.balloons()
            st.success("✅ รอดชีวิตว่ะ! ดวงแข็งจริงๆ")
        else:
            st.error("💀 ไม่รอดว่ะ... เสียใจด้วยนะ")

# --- PAGE: NN THEORY ---
elif page == "🧠 AI vs Real Theory (NN)":
    st.title("🧠 เบื้องหลัง AI แยกรูปจริง/ปลอม")
    st.subheader("📍 แหล่งข้อมูล")
    st.write("ใช้รูปภาพจาก Kaggle (AI vs Real Dataset) มีทั้งภาพถ่ายจริงและภาพที่เจนจาก AI")
    
    st.subheader("🧹 ขั้นตอนการเตรียมรูป")
    st.markdown("""
    1. **Resize:** บีบรูปให้เหลือ 128x128 พิกเซล เพื่อให้โมเดลอ่านง่าย
    2. **Normalize:** เปลี่ยนค่าสีจาก 0-255 ให้เหลือ 0-1 (ช่วยให้เทรนไวขึ้น)
    """)
    
    st.subheader("🧠 อัลกอริทึม: CNN")
    st.write("ใช้ **Convolutional Neural Network** ซึ่งเลียนแบบการมองเห็นของคนเรา สแกนหาจุดที่ AI วาดพลาด!")

# --- PAGE: NN TESTING ---
elif page == "📷 AI vs Real Testing":
    st.title("📷 มาลองเช็คดูว่ารูปนี้ใครวาด?")
    uploaded_file = st.file_uploader("ส่งรูปมาเล้ยย...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB').resize((128, 128))
        st.image(img, caption='รูปที่ส่งมา', width=300)
        with st.spinner('กำลังเล็ง...'):
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = nn_model.predict(img_array)
            score = prediction[0][0]
            label = "🤖 AI เจนฯ มาชัวร์" if score < 0.5 else "📸 ฝีมือคนถ่ายจริงๆ"
            st.subheader(f"ผลทำนาย: {label}")

# --- PAGE: SLEEP HEALTH THEORY & PREDICT (รวมจบในหน้าเดียว) ---
elif page == "💤 Sleep Health Theory & Predict":
    st.title("💤 วิเคราะห์สุขภาพการนอนหลับ")
    
    with st.expander("📖 อ่านรายละเอียดโมเดล (Theory & Cleansing)"):
        st.write("**แหล่งที่มา:** Sleep Health and Lifestyle Dataset จาก Kaggle")
        st.write("**Features:** ดูจากเพศ, อายุ, ชั่วโมงการนอน, ระดับกิจกรรม, ความเครียด และ BMI")
        st.markdown("""
        **การเตรียมข้อมูล (Cleansing):**
        * **จัดการ BMI:** เปลี่ยนกลุ่ม Normal/Obese เป็นตัวเลข 0-3
        * **Label Encoding:** เปลี่ยนชื่อโรคที่เป็นตัวหนังสือให้ AI เข้าใจเป็นกลุ่มตัวเลข
        * **Algorithm:** ใช้ **Random Forest** (สร้างต้นไม้ตัดสินใจหลายๆ ต้นมาช่วยกันตอบ)
        """)

    st.subheader("🧪 มาลองทดสอบสุขภาพคุณดู")
    col1, col2 = st.columns(2)
    with col1:
        s_gender = st.selectbox("เลือกเพศ", ["Male", "Female"])
        s_age = st.number_input("อายุคุณ (ปี)", 10, 80, 30)
        s_dur = st.slider("ชั่วโมงที่นอนต่อวัน", 4.0, 10.0, 7.0)
    with col2:
        # เปลี่ยนจากจำนวนก้าว เป็น ระดับกิจกรรม (0-100)
        s_act = st.slider("ระดับกิจกรรมทางกาย (0-100)", 0, 100, 60)
        s_stress = st.slider("ความเครียด (1-10)", 1, 10, 5)
        s_bmi = st.selectbox("รูปร่างเป็นแบบไหน?", ["Normal", "Normal Weight", "Obese", "Overweight"])
    
    if st.button("ANALYZE SLEEP"):
        bmi_map = {"Normal": 0, "Normal Weight": 1, "Obese": 2, "Overweight": 3}
        gen_val = 0 if s_gender == "Male" else 1
        s_input = np.array([[gen_val, s_age, s_dur, s_act, s_stress, bmi_map[s_bmi]]])
        
        try:
            s_res = sleep_model.predict(s_input)
            st.success(f"### AI บอกว่า: {s_res[0]}")
            if s_res[0] != "None":
                st.warning("⚠️ เฮ้ย! ดูแลสุขภาพหน่อยนะ มีความเสี่ยงเป็นโรคนะเนี่ย")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
