import json
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Biometric Gateway", layout="centered")

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 25px;'>
        <h1 style='color: #0284c7;'>🛡️ Behavioral Biometrics Security Gateway</h1>
        <p style='color: #64748b;'>Connected directly to Colab Isolation Forest Model</p>
    </div>
""",
    unsafe_allow_html=True,
)


# 1. تحميل موديل Colab والميزات الحقيقية
@st.cache_resource
def load_security_system():
  try:
    artifact = joblib.load("biometric_model.pkl")
    return artifact["model"], artifact["features"]
  except Exception as e:
    st.error(f"Error loading model artifact: {e}")
    st.stop()


model, feature_columns = load_security_system()

TARGET_PWD = "Welcome Guest"

# 2. مكون التقاط الأحداث وتوليد كود التوقيت بدقة
html_bridge = f"""
<div style="background-color: #18181b; padding: 25px; border-radius: 12px; border: 1px solid #3f3f46; text-align: center; font-family: sans-serif; color: white;">
    <h3 style="color: #38bdf8; margin-top: 0;">Target Passphrase: <span style="color: #facc15;">{TARGET_PWD}</span></h3>
    <input type="password" id="input_box" autocomplete="off" placeholder="Type passphrase here..." 
           style="width: 90%; padding: 14px; font-size: 18px; border-radius: 8px; border: 2px solid #52525b; background: #27272a; color: white; text-align: center; outline: none; margin-bottom: 15px;">
    <br>
    <p id="guide_msg" style="color: #94a3b8; font-size: 14px; margin: 5px 0;">Type the phrase above. Once finished, the encoded pattern will appear below.</p>
    <textarea id="output_tokens" readonly style="width: 90%; height: 50px; background: #09090b; color: #22c55e; font-family: monospace; font-size: 11px; border: 1px solid #27272a; border-radius: 6px; padding: 6px; text-align: center;"></textarea>
</div>

<script>
    let rawLog = [];
    let keyPresses = {{}};
    const input = document.getElementById('input_box');
    const output = document.getElementById('output_tokens');
    const targetPwd = "{TARGET_PWD}";

    input.addEventListener('keydown', function(e) {{
        if(e.key.length !== 1 && e.key !== 'Backspace') return;
        keyPresses[e.key] = performance.now();
    }});

    input.addEventListener('keyup', function(e) {{
        if(e.key.length !== 1 && e.key !== 'Backspace') return;
        if(keyPresses[e.key]) {{
            rawLog.push({{ key: e.key, down: keyPresses[e.key], up: performance.now() }});
            delete keyPresses[e.key];
        }}

        if(input.value === targetPwd) {{
            let cleanSeq = [];
            let tIndex = targetPwd.length - 1;
            for (let i = rawLog.length - 1; i >= 0; i--) {{
                if (rawLog[i].key === targetPwd[tIndex]) {{
                    cleanSeq.unshift(rawLog[i]);
                    tIndex--;
                    if (tIndex < 0) break;
                }}
            }}

            let attemptData = [];
            for (let i = 0; i < cleanSeq.length; i++) {{
                let hold = (cleanSeq[i].up - cleanSeq[i].down) / 1000.0;
                let flight = (i > 0) ? (cleanSeq[i].down - cleanSeq[i-1].up) / 1000.0 : 0;
                attemptData.push({{ key: cleanSeq[i].key, hold_time: hold, flight_time: flight }});
            }}
            output.value = JSON.stringify(attemptData);
        }} else {{
            output.value = "";
        }}
    }});
</script>
"""

components.html(html_bridge, height=250)

# 3. استلام البيانات ومعالجتها مباشرة في بايثون
payload = st.text_input(
    "Paste or confirm pattern payload to verify:",
    label_visibility="collapsed",
    placeholder="Pattern payload auto-syncs here...",
)

if st.button("🔍 Evaluate Biometric Signature via Colab Model", type="primary"):
  if not payload:
    st.warning(
        "⚠️ Please write 'Welcome Guest' completely in the box above first."
    )
  else:
    try:
      attempt_records = json.loads(payload)
      test_df = pd.DataFrame(attempt_records)

      # استخراج الميزات الحسابية بنفس دوال Colab تماماً
      f_dict = {
          "total_time": (
              test_df["hold_time"].sum() + test_df["flight_time"].sum()
          ),
          "avg_hold": test_df["hold_time"].mean(),
          "avg_flight": test_df["flight_time"].iloc[1:].mean(),
          "std_hold": test_df["hold_time"].std() if len(test_df) > 1 else 0,
          "std_flight": (
              test_df["flight_time"].iloc[1:].std() if len(test_df) > 1 else 0
          ),
      }

      flight_times = test_df["flight_time"].tolist()
      for i in range(1, len(flight_times)):
        f_dict[f"digraph_trans_{i}"] = flight_times[i]

      X_live = pd.DataFrame([f_dict]).fillna(0)

      # مطابقة الأعمدة بدقة مع الموديل
      for col in feature_columns:
        if col not in X_live.columns:
          X_live[col] = 0
      X_live = X_live[feature_columns]

      # قرار نموذج Isolation Forest الحقيقي من Colab
      prediction = model.predict(X_live)[0]
      confidence_score = model.decision_function(X_live)[0]

      st.write("---")
      if prediction == 1:
        st.success("### 🟢 ACCESS GRANTED: Welcome back, Authorized User!")
        st.write(
            "Your digraph timing and keystroke hold patterns match the enrolled"
            " Colab baseline."
        )
      else:
        st.error("### 🔴 ACCESS DENIED: Imposter Detected!")
        st.write(
            "Passphrase text is correct, but your kinetic rhythm was rejected"
            " by the AI security model."
        )

      st.metric(
          label="Colab Isolation Forest Score", value=f"{confidence_score:.4f}"
      )

    except Exception as ex:
      st.error(f"Evaluation error: {ex}")
