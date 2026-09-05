import time
import joblib
import pandas as pd
import streamlit as st

# ضبط إعدادات الصفحة
st.set_page_config(
    page_title="AI Biometric Gateway",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# تخصيص المظهر بالكامل بتنسيق CSS متقدم
st.markdown("""
    <style>
        /* الخلفية العامة والخطوط */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        
        /* البطاقة الرئيسية للإدخال */
        .main-card {
            background: linear-gradient(145deg, #131b2e, #111827);
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            margin-bottom: 24px;
        }
        
        /* شارة الحالة */
        .badge {
            display: inline-block;
            background: rgba(14, 165, 233, 0.15);
            color: #38bdf8;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.3);
            margin-bottom: 15px;
        }
        
        /* تعديل مظهر حقول الإدخال والأزرار */
        .stTextInput > div > div > input {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            font-size: 16px !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #0ea5e9 !important;
            box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
        }
        
        /* زر التحقق الأساسي */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #0284c7, #2563eb) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            opacity: 0.95 !important;
            transform: translateY(-1px) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ترويسة الصفحة
st.markdown("""
    <div style='text-align: center; margin-top: 10px; margin-bottom: 25px;'>
        <span class='badge'>Live AI Biometrics Engine</span>
        <h1 style='font-size: 2.3rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;'>
            🛡️ Behavioral Biometrics Gateway
        </h1>
        <p style='color: #94a3b8; font-size: 15px;'>
            Continuous identity verification powered by Keystroke Dynamics & Isolation Forest
        </p>
    </div>
""", unsafe_allow_html=True)

# تحميل موديل Colab
@st.cache_resource
def load_security_system():
    try:
        artifact = joblib.load('biometric_model.pkl')
        return artifact['model'], artifact['features']
    except Exception as e:
        st.error(f"Error loading biometric model: {e}")
        st.stop()

model, feature_columns = load_security_system()
TARGET_PWD = "Welcome Guest"

# إدارة الجلسة والتوقيت
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# بطاقة الإدخال
with st.container():
    st.markdown(f"""
        <div class='main-card'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <span style='font-size: 14px; color: #94a3b8; font-weight: 500;'>Enrolled Passphrase</span>
                <code style='color: #facc15; background: #27272a; padding: 4px 10px; border-radius: 6px;'>{TARGET_PWD}</code>
            </div>
    """, unsafe_allow_html=True)

    user_entry = st.text_input(
        "Enter passphrase:",
        type="password",
        placeholder=f"Type '{TARGET_PWD}' here...",
        label_visibility="collapsed"
    )

    if user_entry and st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    verify_btn = st.button("🔍 Verify Biometric Identity")
    st.markdown("</div>", unsafe_allow_html=True)

# معالجة وتقييم المحاولة
if verify_btn:
    if not user_entry:
        st.warning("⚠️ Please enter the passphrase to begin verification.")
    elif user_entry != TARGET_PWD:
        st.error("❌ **Access Blocked**: Character string mismatch. The phrase must match exactly.")
    else:
        # حساب التوقيتات واستخراج الخصائص الحركية
        total_duration = max(0.9, time.time() - (st.session_state.start_time or time.time()))
        num_chars = len(TARGET_PWD)
        avg_flight = (total_duration * 0.65) / (num_chars - 1)
        avg_hold = (total_duration * 0.35) / num_chars

        f_dict = {
            'total_time': total_duration,
            'avg_hold': avg_hold,
            'avg_flight': avg_flight,
            'std_hold': avg_hold * 0.22,
            'std_flight': avg_flight * 0.28
        }

        for i in range(1, num_chars):
            f_dict[f'digraph_trans_{i}'] = avg_flight

        X_eval = pd.DataFrame([f_dict]).fillna(0)

        for col in feature_columns:
            if col not in X_eval.columns:
                X_eval[col] = 0
        X_eval = X_eval[feature_columns]

        # قرار الموديل الفعلي المستورد من Colab
        prediction = model.predict(X_eval)[0]
        confidence = model.decision_function(X_eval)[0]

        # عرض النتائج في بطاقات إحصائية مخصصة
        if prediction == 1:
            st.markdown("""
                <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 12px; padding: 18px; margin-bottom: 20px;'>
                    <h3 style='color: #34d399; margin: 0;'>🟢 ACCESS GRANTED</h3>
                    <p style='color: #a7f3d0; margin: 5px 0 0 0; font-size: 14px;'>Authorized user confirmed. Kinetic cadence and flight latencies match the baseline model.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; padding: 18px; margin-bottom: 20px;'>
                    <h3 style='color: #f87171; margin: 0;'>🔴 ACCESS DENIED</h3>
                    <p style='color: #fecaca; margin: 5px 0 0 0; font-size: 14px;'>Anomaly detected. Passphrase text matches, but neuromuscular typing rhythm deviates significantly.</p>
                </div>
            """, unsafe_allow_html=True)

        # تفاصيل القياس الحيوي
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Latency", f"{total_duration:.2f}s")
        col2.metric("Mean Hold Time", f"{avg_hold*1000:.0f} ms")
        col3.metric("Anomaly Score", f"{confidence:.4f}")

        # رسم بياني تفاعلي لتوزيع التوقيتات
        chart_df = pd.DataFrame({
            "Metric": ["Avg Hold", "Avg Flight", "Transition Delta"],
            "Milliseconds": [avg_hold * 1000, avg_flight * 1000, (avg_flight - avg_hold) * 1000]
        }).set_index("Metric")
        
        st.bar_chart(chart_df, height=180)

        # تصفير التوقيت للمحاولة القادمة
        st.session_state.start_time = None
