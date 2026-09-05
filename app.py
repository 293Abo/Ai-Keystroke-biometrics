import json
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Configure application layout and branding
st.set_page_config(page_title="AI Biometric Gateway", layout="centered")

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 25px;'>
        <h1 style='color: #0284c7;'>🛡️ Behavioral Biometrics Security Gateway</h1>
        <p style='color: #64748b;'>AI-driven keystroke dynamics analyzing hold latency and digraph transition rhythms</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Load pre-trained model artifact and required feature signatures
@st.cache_resource
def load_security_system():
  try:
    artifact = joblib.load("biometric-model.pkl")
    return artifact["model"], artifact["features"]
  except Exception as e:
    st.error(f"Error loading model artifact: {e}")
    st.stop()


model, feature_columns = load_security_system()

# Embedded JavaScript component for sub-millisecond hardware event capture
html_component = """
<div style="background-color: #18181b; padding: 25px; border-radius: 12px; border: 1px solid #3f3f46; text-align: center; font-family: sans-serif;">
    <h3 style="color: #38bdf8; margin-top: 0;">Target Passphrase: <span style="color: #facc15;">Welcome Guest</span></h3>
    <input type="password" id="target_box" autocomplete="off" placeholder="Type passphrase and press Enter..." 
           style="width: 90%; padding: 14px; font-size: 18px; border-radius: 8px; border: 2px solid #52525b; background: #27272a; color: white; text-align: center; outline: none; margin-bottom: 15px;">
    <br>
    <button onclick="processAttempt()" style="background-color: #0284c7; color: white; border: none; padding: 12px 30px; font-size: 16px; border-radius: 8px; cursor: pointer; font-weight: bold;">
        🔍 Verify Biometric Signature
    </button>
    <p id="client-feedback" style="margin-top: 15px; font-weight: bold;"></p>
</div>

<script>
    let rawLog = [];
    let keyPresses = {};
    const input = document.getElementById("target_box");
    const feedback = document.getElementById("client-feedback");
    const targetPwd = "Welcome Guest";

    input.addEventListener("keydown", (e) => {
        if(e.key === "Enter") {
            processAttempt();
            return;
        }
        if(e.key.length !== 1) return;
        keyPresses[e.key] = performance.now();
    });

    input.addEventListener("keyup", (e) => {
        if(e.key.length !== 1) return;
        if(keyPresses[e.key]) {
            rawLog.push({ key: e.key, down: keyPresses[e.key], up: performance.now() });
            delete keyPresses[e.key];
        }
    });

    function processAttempt() {
        if(input.value !== targetPwd) {
            feedback.style.color = "#f43f5e";
            feedback.innerText = "❌ Incorrect text! Type 'Welcome Guest' exactly.";
            input.value = "";
            rawLog = [];
            keyPresses = {};
            return;
        }

        feedback.style.color = "#facc15";
        feedback.innerText = "Extracting behavioral biometrics & evaluating...";

        // Trace valid keystrokes backwards to maintain correct timing even after backspaces
        let cleanSeq = [];
        let tIndex = targetPwd.length - 1;
        for (let i = rawLog.length - 1; i >= 0; i--) {
            if (rawLog[i].key === targetPwd[tIndex]) {
                cleanSeq.unshift(rawLog[i]);
                tIndex--;
                if (tIndex < 0) break;
            }
        }

        let currentAttemptData = [];
        for (let i = 0; i < cleanSeq.length; i++) {
            let hold = (cleanSeq[i].up - cleanSeq[i].down) / 1000.0;
            let flight = (i > 0) ? (cleanSeq[i].down - cleanSeq[i-1].up) / 1000.0 : 0;
            currentAttemptData.push({ key: cleanSeq[i].key, hold_time: hold, flight_time: flight });
        }

        // Forward extracted timestamps directly to Python backend
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            value: JSON.stringify(currentAttemptData)
        }, "*");

        input.value = "";
        rawLog = [];
        keyPresses = {};
    }
</script>
"""

payload = components.html(html_component, height=230)

# Process payload received from the browser
if payload:
  try:
    attempt_records = json.loads(payload)
    if attempt_records:
      test_df = pd.DataFrame(attempt_records)

      # 1. Macro-level statistical features
      f_dict = {
          "total_time": (
              test_df["hold_time"].sum() + test_df["flight_time"].sum()
          ),
          "avg_hold": test_df["hold_time"].mean(),
          "avg_flight": test_df["flight_time"].mean(),
          "std_hold": test_df["hold_time"].std() if len(test_df) > 1 else 0,
          "std_flight": (
              test_df["flight_time"].std() if len(test_df) > 1 else 0
          ),
      }

      # 2. Sequential digraph transition features
      flight_times = test_df["flight_time"].tolist()
      for i in range(1, len(flight_times)):
        f_dict[f"digraph_trans_{i}"] = flight_times[i]

      X_live = pd.DataFrame([f_dict]).fillna(0)

      # Realign schema with training baseline
      for col in feature_columns:
        if col not in X_live.columns:
          X_live[col] = 0
      X_live = X_live[feature_columns]

      prediction = model.predict(X_live)[0]
      confidence = model.decision_function(X_live)[0]

      st.write("")
      if prediction == 1:
        st.success("### 🟢 ACCESS GRANTED: Welcome back, Authorized User!")
        st.info(
            "Keystroke latencies and digraph transitions match the enrolled"
            " baseline."
        )
      else:
        st.error("### 🔴 ACCESS DENIED: Imposter Detected!")
        st.warning(
            "Passphrase characters match, but behavioral rhythm deviates"
            " significantly from the owner's baseline."
        )

      st.metric(label="AI Anomaly Decision Score", value=f"{confidence:.4f}")

  except Exception as ex:
    st.error(f"Processing error encountered: {ex}")
