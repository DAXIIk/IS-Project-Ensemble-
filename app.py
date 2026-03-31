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

# --- 1. ส่วนโหลดโมเดล ---
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
    st.write("เว็บนี้รวมโมเดลที่เราตั้งใจทำ ทั้งสายสถิติ (ML) และสายประมวลผลภาพ (NN) รายละเอียดแต่ละตัวอยู่ข้างล่างนี้เลย!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📂 ชุดข้อมูลที่ 1: Titanic\nพยากรณ์การรอดชีวิตจากเหตุการณ์เรือล่ม โดยใช้ข้อมูลพื้นฐานของผู้โดยสาร")
    with col2:
        st.info("### 📂 ชุดข้อมูลที่ 2: Sleep Health\nวิเคราะห์พฤติกรรมการใช้ชีวิต เพื่อประเมินความเสี่ยงโรคที่เกิดจากการนอน")

# --- PAGE: ML TITANIC THEORY ---
elif page == "📊 Titanic Theory (ML)":
    st.title("📊 ทฤษฎีโมเดล Titanic (Ensemble Learning)")
    
    st.subheader("📍 ที่มาของ Dataset")
    st.write("ข้อมูลชุดนี้มาจากเว็บไซต์ **Kaggle** (Titanic: Machine Learning from Disaster) ซึ่งเป็นฐานข้อมูลจริงของเรือไททานิคที่ล่มในปี 1912")
    
    st.subheader("🧬 Features (ปัจจัยที่ใช้พยากรณ์)")
    st.markdown("""
    - **Pclass:** ชั้นที่นั่งผู้โดยสาร (1, 2, 3) สะท้อนถึงฐานะและตำแหน่งห้องพักบนเรือ
    - **Sex:** เพศของผู้โดยสาร (ปัจจัยสำคัญที่สุด)
    - **Age:** อายุ (มีผลต่อการช่วยเหลือเด็กก่อน)
    - **Fare:** ราคาตั๋วเดินทาง
    """)
    
    st.subheader("🧹 การจัดการข้อมูล (Data Cleansing)")
    st.write("เราจัดการข้อมูลที่ไม่สมบูรณ์ด้วยวิธีดังนี้:")
    st.markdown("""
    - **จัดการค่าว่าง (Missing Values):** ในคอลัมน์อายุ (Age) ที่หายไป เราใช้ค่า **Mean (ค่าเฉลี่ย)** มาเติมให้เต็ม
    - **การแปลงข้อมูล (Encoding):** เปลี่ยนข้อมูลประเภทตัวอักษร 'Male/Female' ให้เป็นตัวเลข **0 และ 1** เพื่อให้โมเดลคำนวณได้
    - **Feature Selection:** ตัดข้อมูลที่ไม่เกี่ยวข้องออก เช่น ชื่อผู้โดยสาร และเลขตั๋ว เพราะไม่มีผลทางการสถิติ
    """)
    
    st.subheader("🧠 อัลกอริทึม")
    st.write("ใช้ **Ensemble (Voting Classifier)** ที่รวมเอา Random Forest, XGBoost และ Logistic Regression มาโหวตคำตอบร่วมกัน")

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
    st.title("🧠 ทฤษฎีโมเดลแยกรูปภาพ AI vs Real")
    
    st.subheader("📍 ที่มาของ Dataset")
    st.write("ใช้ชุดรูปภาพจากเว็บไซต์ **Kaggle** (rhythmghai/ai-vs-real-images-dataset) รวบรวมภาพถ่ายจริงเทียบกับภาพที่เจนโดย Midjourney และ Stable Diffusion")
    
    st.subheader("🧬 Features (ปัจจัยที่ใช้พยากรณ์)")
    st.write("โมเดลวิเคราะห์จาก **Pixel Data** และลวดลายของภาพ (Texture) เพื่อหาความผิดปกติที่ AI มักจะทำพลาด เช่น บริเวณขอบภาพหรือความเนียนของแสง")
    
    st.subheader("🧹 การจัดการข้อมูล (Data Cleansing & Preprocessing)")
    st.markdown("""
    - **Resizing:** เนื่องจากรูปมาหลายขนาด เราจึงต้องบีบให้เหลือ **128x128 พิกเซล** เท่ากันหมด
    - **Normalization:** ปรับค่าสี (RGB) จาก 0-255 ให้กลายเป็น **0-1** โดยการหารด้วย 255 เพื่อให้โมเดล Neural Network ประมวลผลได้เสถียรขึ้น
    - **Data Augmentation:** มีการหมุนและพลิกรูปภาพ เพื่อให้โมเดลเก่งขึ้นในการเจอรูปหลายๆ มุม
    """)
    
    st.subheader("🧠 อัลกอริทึม")
    st.write("ใช้ **CNN (Convolutional Neural Network)** ซึ่งออกแบบมาเพื่อเลียนแบบระบบประสาทการมองเห็นของมนุษย์")

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

# --- PAGE: SLEEP HEALTH THEORY & PREDICT ---
elif page == "💤 Sleep Health Theory & Predict":
    st.title("💤 ทฤษฎีวิเคราะห์สุขภาพการนอนหลับ")
    
    with st.expander("📖 อ่านรายละเอียด Dataset & Cleansing (กดเพื่อขยาย)"):
        st.subheader("📍 ที่มาของ Dataset")
        st.write("ข้อมูลมาจาก **Kaggle** (uom190346a/sleep-health-and-lifestyle-dataset) รวบรวมข้อมูลไลฟ์สไตล์สุขภาพของคนหลากหลายอาชีพ")
        
        st.subheader("🧬 Features (ปัจจัยที่ใช้พยากรณ์)")
        st.markdown("""
        - **Gender & Age:** เพศและอายุที่มีผลต่อสภาวะร่างกาย
        - **Sleep Duration:** ชั่วโมงการนอนต่อคืน
        - **Physical Activity Level:** ระดับการขยับร่างกาย (0-100)
        - **Stress Level:** คะแนนความเครียดจากการทำงานและชีวิต
        - **BMI Category:** รูปร่าง (Normal, Overweight, Obese)
        """)
        
        st.subheader("🧹 การจัดการข้อมูล (Data Cleansing)")
        st.markdown("""
        - **Label Encoding:** แปลงกลุ่มอาชีพและ BMI ที่เป็นข้อความให้กลายเป็นตัวเลขกลุ่ม (0, 1, 2, 3)
        - **Handling Missing Data:** คอลัมน์ความผิดปกติทางการนอนที่ว่างอยู่ เราเติมคำว่า **'None'** เพื่อระบุว่าเป็นกลุ่มคนปกติ
        - **Categorical Mapping:** จัดกลุ่ม BMI ที่มีความหมายคล้ายกันให้อยู่ใน Category เดียวกันเพื่อลดความซับซ้อน
        """)
        
        st.subheader("🧠 อัลกอริทึม")
        st.write("ใช้ **Random Forest Classifier** ซึ่งเด่นเรื่องการแตกกิ่งก้านการตัดสินใจ (Decision Trees) ทำให้แม่นยำสูงกับข้อมูลสายสุขภาพ")

    st.divider()
    st.subheader("🧪 มาลองทดสอบสุขภาพคุณดู")
    col1, col2 = st.columns(2)
    with col1:
        s_gender = st.selectbox("เลือกเพศ", ["Male", "Female"])
        s_age = st.number_input("อายุคุณ (ปี)", 10, 80, 30)
        s_dur = st.slider("ชั่วโมงที่นอนต่อวัน", 4.0, 10.0, 7.0)
    with col2:
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
