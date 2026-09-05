from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Dict
import os

app = FastAPI(title="Kinetic Biometrics Realtime API")

# تحميل الموديل تلقائياً من الملفات الموجودة
MODEL_FILES = ['biometric_model (1).pkl', 'biometric_model.pkl']
model = None
features = []

for m_path in MODEL_FILES:
    if os.path.exists(m_path):
        try:
            artifact = joblib.load(m_path)
            model = artifact['model']
            features = artifact['features']
            break
        except Exception:
            continue

class BiometricPayload(BaseModel):
    features: Dict[str, float]

@app.post("/api/verify")
def verify_keystrokes(payload: BiometricPayload):
    if model is None:
        return {"authorized": False, "score": 0.0, "dwell_ratio": 0.0, "error": "Model not loaded"}

    # بناء مصفوفة الميزات الحقيقية المستخرجة من أجهزة المستخدم
    df = pd.DataFrame([payload.features])
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    df = df[features]

    # استدعاء دالة القرار لـ One-Class SVM الحقيقي
    prediction = int(model.predict(df)[0])
    raw_score = float(model.decision_function(df)[0])

    # قبول المالك إذا كان في النطاق الطبيعي للنموذج المدرب
    is_authorized = (prediction == 1) or (raw_score >= -0.25)

    return {
        "authorized": is_authorized,
        "score": round(raw_score, 4),
        "dwell_ratio": round(payload.features.get("dwell_ratio", 0.0), 3)
    }

@app.get("/", response_class=HTMLResponse)
def serve_portal():
    with open("portal.html", "r", encoding="utf-8") as f:
        return f.read()
