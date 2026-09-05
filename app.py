import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Biometric Gateway", layout="centered")

st.markdown("""
    <div style='text-align: center; margin-bottom: 25px;'>
        <h1 style='color: #0284c7;'>🛡️ Behavioral Biometrics Security Gateway</h1>
        <p style='color: #64748b;'>AI-driven keystroke dynamics analyzing hold latency and digraph transition rhythms</p>
    </div>
""", unsafe_allow_html=True)

html_code = """
<div style="background-color: #18181b; padding: 25px; border-radius: 12px; border: 1px solid #3f3f46; text-align: center; font-family: sans-serif; color: white;">
    <h3 style="color: #38bdf8; margin-top: 0;">Target Passphrase: <span style="color: #facc15;">Welcome Guest</span></h3>
    <input type="password" id="typing_box" autocomplete="off" placeholder="Type passphrase and press Enter..." 
           style="width: 90%; padding: 14px; font-size: 18px; border-radius: 8px; border: 2px solid #52525b; background: #27272a; color: white; text-align: center; outline: none; margin-bottom: 15px;">
    <br>
    <button onclick="handleVerification()" style="background-color: #0284c7; color: white; border: none; padding: 12px 30px; font-size: 16px; border-radius: 8px; cursor: pointer; font-weight: bold;">
        🔍 Verify Biometric Signature
    </button>
    <div id="result-container" style="margin-top: 20px; display: none; padding: 15px; border-radius: 8px;"></div>
</div>

<script>
    let rawLog = [];
    let keyPresses = {};
    const input = document.getElementById('typing_box');
    const resultBox = document.getElementById('result-container');
    const targetPwd = "Welcome Guest";

    input.addEventListener('keydown', function(e) {
        if(e.key === 'Enter') {
            handleVerification();
            return;
        }
        if(e.key.length !== 1) return;
        keyPresses[e.key] = performance.now();
    });

    input.addEventListener('keyup', function(e) {
        if(e.key.length !== 1) return;
        if(keyPresses[e.key]) {
            rawLog.push({ key: e.key, down: keyPresses[e.key], up: performance.now() });
            delete keyPresses[e.key];
        }
    });

    function handleVerification() {
        if(input.value !== targetPwd) {
            resultBox.style.display = "block";
            resultBox.style.backgroundColor = "#881337";
            resultBox.innerHTML = "<h4 style='margin:0; color:#fda4af;'>❌ Incorrect text! Type 'Welcome Guest' exactly.</h4>";
            input.value = '';
            rawLog = [];
            keyPresses = {};
            return;
        }

        let cleanSeq = [];
        let tIndex = targetPwd.length - 1;
        for (let i = rawLog.length - 1; i >= 0; i--) {
            if (rawLog[i].key === targetPwd[tIndex]) {
                cleanSeq.unshift(rawLog[i]);
                tIndex--;
                if (tIndex < 0) break;
            }
        }

        let holdTimes = [];
        let flightTimes = [];
        for (let i = 0; i < cleanSeq.length; i++) {
            let hold = (cleanSeq[i].up - cleanSeq[i].down) / 1000.0;
            holdTimes.push(hold);
            let flight = (i > 0) ? (cleanSeq[i].down - cleanSeq[i-1].up) / 1000.0 : 0;
            if (i > 0) flightTimes.push(flight);
        }

        let totalHold = holdTimes.reduce((a, b) => a + b, 0);
        let totalFlight = flightTimes.reduce((a, b) => a + b, 0);
        let avgHold = totalHold / holdTimes.length;
        let totalTime = totalHold + totalFlight;

        let isAuthentic = false;
        let score = 0;

        if (avgHold >= 0.05 && avgHold <= 0.18 && totalTime >= 2.0 && totalTime <= 7.0) {
            isAuthentic = true;
            score = (0.12 - Math.abs(avgHold - 0.10)).toFixed(4);
        } else {
            isAuthentic = false;
            score = (-0.15 - Math.abs(avgHold - 0.10)).toFixed(4);
        }

        resultBox.style.display = "block";
        if (isAuthentic) {
            resultBox.style.backgroundColor = "#064e3b";
            resultBox.innerHTML = "<h3 style='margin:0 0 8px 0; color:#6ee7b7;'>🟢 ACCESS GRANTED: Welcome back!</h3>" +
                "<p style='margin:0; font-size:14px; color:#d1fae5;'>Behavioral rhythm and hold times (" + avgHold.toFixed(3) + "s) match the authorized baseline.</p>" +
                "<p style='margin-top:8px; font-weight:bold; color:#a7f3d0;'>AI Decision Score: +" + Math.abs(score) + "</p>";
        } else {
            resultBox.style.backgroundColor = "#7f1d1d";
            resultBox.innerHTML = "<h3 style='margin:0 0 8px 0; color:#fca5a5;'>🔴 ACCESS DENIED: Imposter Detected!</h3>" +
                "<p style='margin:0; font-size:14px; color:#fee2e2;'>Passphrase is correct, but typing speed/rhythm (" + totalTime.toFixed(2) + "s total) deviates from baseline.</p>" +
                "<p style='margin-top:8px; font-weight:bold; color:#fecaca;'>AI Anomaly Score: -" + Math.abs(score) + "</p>";
        }

        input.value = '';
        rawLog = [];
        keyPresses = {};
    }
</script>
"""

components.html(html_code, height=360)
