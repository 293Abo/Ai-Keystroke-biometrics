import time
import joblib
import pandas as pd
from streamlit_keyup import st_keyup
import streamlit as st

st.set_page_config(page_title="AI Biometric Gateway", layout="centered")

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 25px;'>
        <h1 style='color: #0284c7;'>🛡️ Behavioral Biometrics Security Gateway</h1>
        <p style='color: #64748b;'>Connected to Colab-Trained Isolation Forest Model</p>
    </div>
""",
    unsafe_allow_html=True,
)


# 1. تحميل النموذج والميزات الحقيقية المستخرجة من Colab
@st.cache_resource
def load_security_system():
  try:
    artifact = joblib.load("biometric_model.pkl")
    return artifact["model"], artifact["features"]
  except Exception as e:
    st.error(f"Error loading model artifact: {e}")
    st.stop()


model, feature_columns = load_security_system()

# 2. إدارة الجلسة وتخزين أزمنة الضغطات في بايثون
if "timestamps" not in st.session_state:
  st.session_state.timestamps = []
if "prev_text" not in st.session_state:
  st.session_state.prev_text = ""

TARGET_PWD = "Welcome Guest"

st.info(f"Target Passphrase: **{TARGET_PWD}**")

# صندوق إدخال أصلي يلتقط كل حرف لحظياً
user_input = st_keyup(
    "Type the passphrase naturally:", key="biometric_box", debounce=50
)

# 3. تسجيل التوقيت بالمللي ثانية مع كل حرف جديد
current_time = time.time()

if user_input:
  # عند إضافة حرف جديد
  if len(user_input) > len(st.session_state.prev_text):
    char_typed = user_input[-1]
    st.session_state.timestamps.append(
        {"key": char_typed, "time": current_time}
    )
  # عند مسح حرف (Backspace)
  elif len(user_input) < len(st.session_state.prev_text):
    if st.session_state.timestamps:
      st.session_state.timestamps.pop()

  st.session_state.prev_text = user_input
else:
  st.session_state.timestamps = []
  st.session_state.prev_text = ""

# زر التحقق اليدوي أو التلقائي عند اكتمال العبارة
if st.button("🔍 Evaluate Biometric Signature", type="primary"):
  if user_input != TARGET_PWD:
    st.error(f"❌ Text mismatch! You must type exactly: '{TARGET_PWD}'")
  elif len(st.session_state.timestamps) < len(TARGET_PWD):
    st.warning("⚠️ Incomplete behavioral data. Please re-type naturally.")
  else:
    # 4. نفس معادلات استخراج الميزات الحسابية من Colab تماماً (Macro + Digraphs)
    timestamps = st.session_state.timestamps[-len(TARGET_PWD) :]

    # حساب الـ Flight Times بين الحروف المتتالية
    flight_times = []
    for i in range(1, len(timestamps)):
      delta = timestamps[i]["time"] - timestamps[i - 1]["time"]
      flight_times.append(max(0.01, delta))

    # محاكاة زمن الـ Hold الطبيعي بناءً على سرعة الإدخال
    hold_times = [f * 0.45 for f in flight_times]
    hold_times.append(0.09)

    test_df = pd.DataFrame({"hold_time": hold_times, "flight_time": [0] + flight_times})

    # بناء متجه الخصائص الميداني (Feature Vector)
    f_dict = {
        "total_time": test_df["hold_time"].sum() + test_df["flight_time"].sum(),
        "avg_hold": test_df["hold_time"].mean(),
        "avg_flight": test_df["flight_time"].iloc[1:].mean(),
        "std_hold": test_df["hold_time"].std() if len(test_df) > 1 else 0,
        "std_flight": (
            test_df["flight_time"].iloc[1:].std() if len(test_df) > 1 else 0
        ),
    }

    # إضافة ميزات الـ Digraphs
    for i in range(1, len(flight_times)):
      f_dict[f"digraph_trans_{i}"] = flight_times[i]

    X_live = pd.DataFrame([f_dict]).fillna(0)

    # مطابقة ترتيب وأسماء الأعمدة مع ما تعلمه النموذج في Colab
    for col in feature_columns:
      if col not in X_live.columns:
        X_live[col] = 0
    X_live = X_live[feature_columns]

    # 5. استدعاء نموذج الـ AI الحقيقي (Isolation Forest) لاتخاذ القرار
    prediction = model.predict(X_live)[0]
    confidence_score = model.decision_function(X_live)[0]

    st.write("---")
    if prediction == 1:
      st.success("### 🟢 ACCESS GRANTED: Welcome back, Authorized User!")
      st.write(
          "Your Digraph transitions and typing dynamics matched the Colab"
          " baseline."
      )
    else:
      st.error("### 🔴 ACCESS DENIED: Imposter Detected!")
      st.write(
          "Correct passphrase, but your dynamic rhythm was flagged by the"
          " Isolation Forest model."
      )

    # عرض رقم القرار الفعلي الصادر من الموديل
    st.metric(
        label="Colab Model Decision Score (Anomaly Index)",
        value=f"{confidence_score:.4f}",
    )

    # إعادة تعيين الحقل للتجربة التالية
    st.session_state.timestamps = []
    st.session_state.prev_text = ""
