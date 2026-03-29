import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

# --- ตั้งค่าหน้าเว็บแบบคูลๆ ---
st.set_page_config(page_title="AI Project Hub", page_icon="🚀", layout="wide")

# ใส่ CSS แต่งโทน Dark Mode ให้ดู Modern
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-image: linear-gradient(to right, #1e3a8a, #3b82f6); 
        color: white; border: none; font-weight: bold;
    }
    .stButton>button:hover { opacity: 0.8; color: white; }
    h1, h2, h3 { color: #60a5fa; font-family: 'Kanit', sans-serif; }
    .stSelectbox, .stSlider, .stNumberInput { border-radius: 10px; }
    .sidebar .sidebar-content { background-color: #111827; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. โหลดโมเดล (ใช้ cache จะได้ไม่โหลดซ้ำให้ช้า) ---
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

# --- 2. เมนู Sidebar ---
st.sidebar.markdown("# 🛠️ AI Dashboard")
page = st.sidebar.radio("อยากไปหน้าไหนดี?", [
    "🏠 หน้าแรก (Overview)",
    "🚢 ทฤษฎีเรือไททานิค (ML)", 
    "🔮 ลองทายดูว่าจะรอดไหม?",
    "🤖 ทฤษฎี AI vs Real (NN)",
    "🖼️ ลองส่งรูปมาตรวจดู",
    "💤 แถม: วิเคราะห์การนอน"
])

# --- หน้าแรก: OVERVIEW ---
if page == "🏠 หน้าแรก (Overview)":
    st.title("🚀 ยินดีต้อนรับเข้าสู่ AI Project!")
    st.write("เว็บนี้รวมโมเดลที่เราปั้นมากับมือ ทั้งสายสถิติ (ML) และสายเจ๋งๆ อย่างภาพ (NN) มาลองเล่นกันได้เลยครับ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📦 Dataset 1: Titanic\nข้อมูลดิบจาก Kaggle เอามาทำนายว่าใครจะรอดจากเหตุการณ์เรือล่มบ้าง โดยใช้สถิติต่างๆ")
    with col2:
        st.info("### 📦 Dataset 2: AI vs Real\nใช้รูปภาพมาเทรนให้ AI แยกให้ออกว่า รูปไหนคนถ่ายจริงๆ รูปไหน AI เจนฯ ขึ้นมา")

# --- หน้า ML THEORY ---
elif page == "🚢 ทฤษฎีเรือไททานิค (ML)":
    st.title("🚢 เจาะลึกโมเดล Titanic (Ensemble)")
    st.subheader("📍 ข้อมูลมาจากไหน?")
    st.write("ดึงมาจากเว็บ Kaggle (Titanic: ML from Disaster) เป็นข้อมูลผู้โดยสารที่มีทั้งชื่อ อายุ เพศ และราคาตั๋ว")
    
    st.subheader("🧹 ทำความสะอาดข้อมูล (Cleansing)")
    st.write("ตอนแรกข้อมูลมันเละนิดหน่อยครับ เราเลยต้องแก้แบบนี้:")
    st.markdown("""
    - **จัดการช่องว่าง:** อายุ (Age) ใครไม่ระบุ เราใส่ค่าเฉลี่ยลงไปแทน
    - **เปลี่ยนตัวหนังสือเป็นเลข:** เพศ (Sex) ชาย/หญิง AI อ่านไม่ออก เลยเปลี่ยนเป็น 1 กับ 0 ซะเลย
    - **คัดเฉพาะเน้นๆ:** พวกเลขตั๋วหรือเลขห้องพักที่ดูไม่ช่วยอะไร เราโยนทิ้งไปเลย (Drop)
    """)
    
    st.subheader("🧠 ใช้ท่าไหนเทรน? (Ensemble)")
    st.write("เราไม่ได้ใช้โมเดลเดียวเสี่ยงดวง แต่ใช้ **Voting Classifier** เอา 3 ตัวท็อป (Random Forest, XGBoost, Logistic Regression) มาโหวตกัน ใครชนะก็ตอบอันนั้น!")

# --- หน้า ML TESTING ---
elif page == "🔮 ลองทายดูว่าจะรอดไหม?":
    st.title("🔮 มาลองทำนายกันว่าจะรอดไหม?")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("เลือกชั้นที่นั่ง (1=รวยสุด, 3=ประหยัด)", [1, 2, 3])
            sex = st.radio("เลือกเพศ", ["Male", "Female"])
            age = st.slider("ใส่อายุเท่าไหร่ดี?", 0, 80, 25)
        with col2:
            fare = st.number_input("ราคาตั๋ว (ยิ่งแพงยิ่งมีโอกาสรอดนะ)", 0.0, 500.0, 32.0)
    
    if st.button("กดเพื่อทำนายผล"):
        sex_val = 1 if sex == "Male" else 0
        input_data = np.array([[pclass, sex_val, age, 0, 0, fare, 0]])
        res = ml_model.predict(input_data)
        
        if res[0] == 1:
            st.balloons()
            st.success("🎉 ยินดีด้วย! คุณมีโอกาสรอดสูงมาก")
        else:
            st.error("💀 เสียใจด้วย... โอกาสรอดยากนิดนึงนะ")

# --- หน้า NN THEORY ---
elif page == "🤖 ทฤษฎี AI vs Real (NN)":
    st.title("🤖 เบื้องหลังการแยกรูป AI vs Real")
    st.subheader("📍 แหล่งข้อมูล")
    st.write("ใช้ชุดรูปภาพจาก Kaggle (AI-Generated vs Real Images) มีทั้งรูปวิว รูปคน รูปสัตว์")
    
    st.subheader("🧹 ขั้นตอนการเตรียมรูปภาพ")
    st.markdown("""
    1. **ปรับขนาด:** รูปมาเล็กบ้างใหญ่บ้าง เราจับยืด/หดให้เหลือ 128x128 เท่ากันหมด
    2. **ปรับสี:** เปลี่ยนค่าพิกเซลจาก 0-255 ให้เหลือ 0-1 (Normalization) ช่วยให้ AI เรียนไวขึ้น
    """)
    
    st.subheader("🧠 อัลกอริทึม (Neural Network)")
    st.write("ใช้ **CNN (Convolutional Neural Network)** ซึ่งเก่งเรื่องจำลวดลายภาพมาก เหมือนจำว่าเส้นแบบนี้ AI วาด หรือเส้นแบบนี้กล้องถ่าย")

# --- หน้า NN TESTING ---
elif page == "🖼️ ลองส่งรูปมาตรวจดู":
    st.title("🖼️ ตรวจสอบรูปภาพ")
    uploaded_file = st.file_uploader("ส่งรูปมาเล้ยย (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB').resize((128, 128))
        st.image(img, caption='รูปที่คุณส่งมา', width=300)
        
        with st.spinner('แป๊บนะ AI กำลังเล็ง...'):
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = nn_model.predict(img_array)
            score = prediction[0][0]
            label = "🤖 AI วาดแน่ๆ!" if score < 0.5 else "📸 ฝีมือคนถ่ายจริงๆ"
            
            st.subheader(f"ผลออกมาคือ: {label}")
            st.write(f"ความมั่นใจ: {score:.4f}")

# --- หน้า SLEEP HEALTH ---
elif page == "💤 แถม: วิเคราะห์การนอน":
    st.title("💤 วิเคราะห์สุขภาพการนอน")
    st.write("แถมให้ครับ! ใช้ข้อมูลพฤติกรรมมาดูว่าการนอนของคุณปกติไหม?")
    
    col1, col2 = st.columns(2)
    with col1:
        s_gender = st.selectbox("เลือกเพศ", ["Male", "Female"])
        s_age = st.number_input("อายุคุณเท่าไหร่?", 10, 80, 30)
        s_dur = st.slider("นอนวันละกี่ชั่วโมง?", 4.0, 10.0, 7.0)
    with col2:
        s_act = st.number_input("เดินวันละกี่ก้าว?", 0, 20000, 5000)
        s_stress = st.slider("เครียดแค่ไหน (1-10)?", 1, 10, 5)
        s_bmi = st.selectbox("รูปร่างเป็นยังไง?", ["Normal", "Normal Weight", "Obese", "Overweight"])
    
    if st.button("วิเคราะห์สุขภาพ"):
        bmi_map = {"Normal": 0, "Normal Weight": 1, "Obese": 2, "Overweight": 3}
        gen_val = 0 if s_gender == "Male" else 1
        s_input = np.array([[gen_val, s_age, s_dur, s_act, s_stress, bmi_map[s_bmi]]])
        s_res = sleep_model.predict(s_input)
        
        st.success(f"📊 AI บอกว่า: {s_res[0]}")
        if s_res[0] != "None":
            st.warning("ดูแลสุขภาพด้วยนะ เป็นห่วงนะเนี่ย!")
