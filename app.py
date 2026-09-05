import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from streamlit_javascript import st_javascript

st.set_page_config(page_title="AI Biometric Gateway", layout="centered")

st.markdown("""
    <div style='text-align: center; margin-bottom: 25px;'>
        <h1 style='color: #0284c7;'>🛡️ Behavioral Biometrics Security Gateway</h1>
        <p style='color: #64748b;'>AI-driven keystroke dynamics analyzing hold latency and digraph transition rhythms</p>
    </div>
""", unsafe_allow_html=True)

# Load model artifact
@st.cache_resource
def load_security_system():
    try:
        artifact = joblib.load('biometric_model.pkl')
        return artifact['model'], artifact['features']
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        st.stop()

model, feature_columns = load_security_system()

# Embed Self-Contained HTML Form with localStorage Bridge
st.components.v1.html("""
<div style="background-color: #18181b; padding: 25px; border-radius: 12px; border: 1px solid #3f3f46; text-align: center; font-family: sans-serif;">
    <h3 style="color: #38bdf8; margin-top: 0;">Target Passphrase: <span style="color: #facc15;">Welcome Guest</span></h3>
    <input type="password" id="typing_box" autocomplete="off" placeholder="Type passphrase and press Enter..." 
           style="width: 90%; padding: 14px; font-size: 18px; border-radius: 8px; border: 2px solid #52525b; background: #27272a; color: white; text-align: center; outline: none; margin-bottom: 15px;">
    <br>
    <button onclick="handleVerification()" style="background-color: #0284c7; color: white; border: none; padding: 12px 30px; font-size: 16px; border-radius: 8px; cursor: pointer; font-weight: bold;">
        🔍 Verify Biometric Signature
    </button>
    <p id="feedback-msg" style="margin-top: 15px; font-weight: bold;"></p>
</div>

<script>
    let rawLog = [];
    let keyPresses = {};
    const input = document.getElementById('typing_box');
    const feedback = document.getElementById('feedback-msg');
    const targetPwd = "Welcome Guest";

    input.addEventListener('keydown', (e) => {
        if(e.key === 'Enter') {
            handleVerification();
            return;
        }
        if(e.key.length !== 1) return;
        keyPresses[e.key] = performance.now();
    });

    input.addEventListener('keyup', (e) => {
        if(e.key.length !== 1) return;
        if(keyPresses[e.key]) {
            rawLog.push({ key: e.key, down: keyPresses[e.key], up: performance.now() });
            delete keyPresses[e.key];
        }
    });

    function handleVerification() {
        if(input.value !== targetPwd) {
            feedback.style.color = "#f43f5e";
            feedback.innerText = "❌ Incorrect text! Type 'Welcome Guest' exactly.";
            input.value = ''; rawLog = []; keyPresses = {};
            return;
        }

        feedback.style.color = "#22c55e";
        feedback.innerText = "✅ Pattern captured! Processing decision...";

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

        // Save safely into browser storage shared across iframes
        window.localStorage.setItem("latest_biometric_data", JSON.stringify(currentAttemptData));
        window.localStorage.setItem("biometric_timestamp", Date.now().toString());

        input.value = '';
        rawLog = [];
        keyPresses = {};
    }
</script>
""", height=220)

# Retrieve data directly using streamlit_javascript (Bypasses all iframe security restrictions)
retrieved_json = st_javascript("""await (async () => {
    return window.localStorage.getItem("latest_biometric_data");
})()""")

if retrieved_json and retrieved_json != "null":
    try:
        attempt_records = json.loads(retrieved_json)
        if attempt_records and len(attempt_records) > 0:
            test_df = pd.DataFrame(attempt_records)

            f_dict = {
                'total_time': test_df['hold_time'].sum() + test_df['flight_time'].sum(),
                'avg_hold': test_df['hold_time'].mean(),
                'avg_flight': test_df['flight_time'].mean(),
                'std_hold': test_df['hold_time'].std() if len(test_df) > 1 else 0,
                'std_flight': test_df['flight_time'].std() if len(test_df) > 1 else 0
            }

            flight_times = test_df['flight_time'].tolist()
            for i in range(1, len(flight_times)):
                f_dict[f'digraph_trans_{i}'] = flight_times[i]

            X_live = pd.DataFrame([f_dict]).fillna(0)

            for col in feature_columns:
                if col not in X_live.columns:
                    X_live[col] = 0
            X_live = X_live[feature_columns]

            prediction = model.predict(X_live)[0]
            confidence = model.decision_function(X_live)[0]

            st.write("")
            if prediction == 1:
                st.success("### 🟢 ACCESS GRANTED: Welcome back, Authorized User!")
                st.info("Keystroke latencies and digraph transitions match the enrolled baseline.")
            else:
                st.error("### 🔴 ACCESS DENIED: Imposter Detected!")
                st.warning("Passphrase characters match, but behavioral rhythm deviates significantly from the owner's baseline.")

            st.metric(label="AI Anomaly Decision Score", value=f"{confidence:.4f}")

            # Clear button to allow another test
            if st.button("Reset / Test Again"):
                st_javascript("""window.localStorage.removeItem("latest_biometric_data");""")
                st.rerun()

    except Exception as ex:
        st.error(f"Evaluation error: {ex}")
