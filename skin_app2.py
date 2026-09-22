from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import joblib
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

app = Flask(__name__)

# ==========================================================
# 1. LOAD TRAINED MODELS & WEIGHTS AT SERVER STARTUP
# ==========================================================

rf_model = joblib.load('rf_model.pkl')
class_names = joblib.load('class_names.pkl')

weights = EfficientNet_B0_Weights.DEFAULT
efficientnet = models.efficientnet_b0(weights=weights)
efficientnet.classifier = nn.Identity()
efficientnet.eval()

transform = weights.transforms()

# ==========================================================
# 2. ADVANCED GLASSMORPHISM & 3D ANIMATED HTML TEMPLATE
# ==========================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dermatology AI Diagnostics</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        body {
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #0f172a;
            overflow-x: hidden;
            perspective: 1000px;
            padding: 20px;
        }

        /* Ambient Liquid Glass Animation Blobs */
        .liquid-blob {
            position: absolute;
            filter: blur(80px);
            border-radius: 50%;
            z-index: 0;
            animation: float 10s infinite ease-in-out alternate;
        }

        .blob-1 {
            width: 350px;
            height: 350px;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            top: 10%;
            left: 15%;
        }

        .blob-2 {
            width: 400px;
            height: 400px;
            background: linear-gradient(135deg, #0ea5e9, #3b82f6);
            bottom: 10%;
            right: 15%;
            animation-delay: -5s;
        }

        @keyframes float {
            0% { transform: translate(0px, 0px) scale(1) rotate(0deg); }
            50% { transform: translate(40px, -50px) scale(1.1) rotate(180deg); }
            100% { transform: translate(-30px, 30px) scale(0.95) rotate(360deg); }
        }

        /* 3D Glassmorphism Container */
        .glass-card {
            position: relative;
            z-index: 10;
            width: 100%;
            max-width: 580px;
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4),
                        inset 0 1px 1px rgba(255, 255, 255, 0.3);
            transform-style: preserve-3d;
            transition: transform 0.5s cubic-bezier(0.23, 1, 0.32, 1), box-shadow 0.5s ease;
            animation: slideUp 0.8s ease-out;
        }

        .glass-card:hover {
            transform: translateY(-6px) rotateX(2deg) rotateY(-2deg);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5),
                        inset 0 1px 2px rgba(255, 255, 255, 0.5);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px) scale(0.95); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        h2 {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 25px;
            text-align: center;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            color: #e2e8f0;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 8px;
            letter-spacing: 0.3px;
        }

        /* Glass Form Controls */
        input[type="number"], select, input[type="file"] {
            width: 100%;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        select option {
            background: #1e293b;
            color: #ffffff;
        }

        input[type="number"]:focus, select:focus, input[type="file"]:focus {
            background: rgba(255, 255, 255, 0.12);
            border-color: #38bdf8;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
        }

        /* Custom Animated File Input Styling */
        input[type="file"]::file-selector-button {
            margin-right: 12px;
            padding: 8px 16px;
            border-radius: 8px;
            border: none;
            background: rgba(255, 255, 255, 0.2);
            color: #ffffff;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        input[type="file"]::file-selector-button:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        /* Modern 3D Action Button */
        .btn-submit {
            width: 100%;
            padding: 14px;
            margin-top: 10px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 8px 20px rgba(56, 189, 248, 0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 25px rgba(56, 189, 248, 0.5);
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Results Box Styling */
        .results-box {
            margin-top: 30px;
            padding-top: 25px;
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            animation: fadeIn 0.6s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .results-box h3 {
            color: #ffffff;
            font-size: 1.25rem;
            margin-bottom: 15px;
        }

        .result-item {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .badge {
            padding: 6px 14px;
            border-radius: 20px;
            color: #ffffff;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .advice-box {
            margin-top: 15px;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.05);
            border-left: 4px solid #38bdf8;
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 0.9rem;
            line-height: 1.5;
        }
    </style>
</head>
<body>

    <!-- Ambient Liquid Glass Background Blobs -->
    <div class="liquid-blob blob-1"></div>
    <div class="liquid-blob blob-2"></div>

    <!-- Glassmorphism Container Card -->
    <div class="glass-card">
        <h2>Dermatology Disease Classifier</h2>
        
        <form action="/predict" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label>Upload Lesion Image</label>
                <input type="file" name="image" accept="image/*" required>
            </div>

            <div class="form-group">
                <label>Patient Age</label>
                <input type="number" name="age" min="1" max="100" value="30" required>
            </div>

            <div class="form-group">
                <label>Patient Gender</label>
                <select name="gender">
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <button type="submit" class="btn-submit">Predict Condition</button>
        </form>

        {% if predicted_class %}
        <div class="results-box">
            <h3>Prediction Results</h3>
            <div class="result-item">
                <span>Detected Condition:</span>
                <strong style="color: #ffffff;">{{ predicted_class }}</strong>
            </div>
            <div class="result-item">
                <span>Confidence Score:</span>
                <strong style="color: #38bdf8;">{{ confidence }}%</strong>
            </div>
            <div class="result-item">
                <span>Urgency Level:</span>
                <span class="badge" style="background-color: {{ badge_color }};">{{ urgency_level }}</span>
            </div>
            <div class="advice-box">
                <strong>Doctor Advice:</strong> {{ advice }}
            </div>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

# ==========================================================
# 3. ROUTE HANDLERS
# ==========================================================

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    age = float(request.form['age'])
    gender = float(request.form['gender'])

    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        image_features = efficientnet(image_tensor).numpy()[0]

    meta_features = np.array([age / 100.0, gender], dtype=np.float32)
    combined_input = np.concatenate((image_features, meta_features)).reshape(1, -1)

    prediction_idx = rf_model.predict(combined_input)[0]
    predicted_class = class_names[prediction_idx]
    confidence = round(np.max(rf_model.predict_proba(combined_input)[0]) * 100, 2)

    high_urgency = [
        'Melanoma Skin Cancer Nevi and Moles',
        'Bullous Disease Photos',
        'Cellulitis Impetigo and other Bacterial Infections',
        'Vascular Tumors'
    ]
    
    moderate_urgency = [
        'Exanthems and Drug Eruptions',
        'Herpes HPV and other STDs Photos',
        'Psoriasis pictures Lichen Planus and related diseases',
        'Scabies Lyme Disease and other Infestations and Bites',
        'Eczema Photos',
        'Tinea Ringworm Candidiasis and other Fungal Infections'
    ]

    if predicted_class in high_urgency:
        urgency_level = "HIGH URGENCY"
        badge_color = "#ef4444"  # Modern Vibrant Red
        advice = "This condition requires immediate evaluation by a dermatologist or healthcare provider."
    elif predicted_class in moderate_urgency:
        urgency_level = "MODERATE URGENCY"
        badge_color = "#f97316"  # Modern Vibrant Orange
        advice = "Please schedule a consultation with a dermatologist for timely clinical assessment and treatment."
    elif predicted_class.lower() == 'normal':
        urgency_level = "NON-PATHOLOGICAL"
        badge_color = "#10b981"  # Modern Vibrant Green
        advice = "No skin abnormalities detected. Consult a doctor if you experience pain or unexpected changes."
    else:
        urgency_level = "LOW URGENCY"
        badge_color = "#64748b"  # Modern Slate Gray
        advice = "Non-urgent condition. Consider consulting a medical professional if symptoms persist or bother you."

    return render_template_string(
        HTML_TEMPLATE,
        predicted_class=predicted_class,
        confidence=confidence,
        urgency_level=urgency_level,
        badge_color=badge_color,
        advice=advice
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)