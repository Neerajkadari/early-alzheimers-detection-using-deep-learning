🧠 Alzheimer’s MRI Screening System (AI)

An AI-powered web application for early screening of Alzheimer’s disease using brain MRI images.
The system is designed as a two-stage deep learning pipeline to prioritize early risk detection while maintaining stable and interpretable predictions.

⚠️ This project is intended for research and screening support only and is not a medical diagnosis tool.

🚀 Overview

Early identification of cognitive impairment is critical for timely clinical intervention.
This project demonstrates how deep learning models can be integrated into a real-world screening workflow, combining:

Conservative screening logic

Risk escalation

Clear user feedback

Practical deployment constraints

The application is built using TensorFlow and Flask, with a focus on robust inference behavior, not just raw accuracy.

🧠 System Architecture
🔹 Stage 1 — Screening Model

Classifies MRI scans into:

Normal (CN)

Cognitive risk (MCI / AD)

Uses a confidence threshold to decide whether further analysis is required

🔹 Stage 2 — Confirmation Model

Activated only if Stage 1 detects risk

Differentiates between:

Early Cognitive Impairment (Early AD)

Advanced Cognitive Impairment (AD risk)

This two-stage design mirrors real medical screening pipelines:

Reduce false negatives

Escalate uncertain cases

Avoid overconfident single-step predictions

✨ Key Features

🧠 MRI image upload via web interface

🔍 Two-stage deep learning inference

🧩 Backend-only probability logic

🎯 Clear final risk classification

🔔 Audio alerts for normal vs risk cases

🏥 Clean, hospital-style user interface

📦 Single-file Flask application (easy to deploy)

🛠️ Technology Stack

Python

TensorFlow / Keras

Flask

NumPy

Pillow (PIL)

 📥 Trained Models

Due to GitHub file size limits, trained model files are provided via
GitHub Releases.

Download
Go to the Releases section of this repository and download:
- `alzheimers_oasis_early_ad.keras`
- `alzheimer_cnn_v2.keras`

Place the models locally before running the application.
