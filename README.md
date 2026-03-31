<div align="center">
  
# 🚀 Predicting Startup Acquisition
  
**Machine Learning Pipeline & Web Application for Dual-Model Startup Success Prediction**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<br>

## 📖 Overview

This repository contains a robust, end-to-end **Machine Learning Pipeline Web Application** designed to predict the acquisition status of startups. Powered by a Flask backend and Scikit-Learn machine learning algorithms, this solution supports both **binary** and **multiclass** classification to provide intelligent, data-driven predictions.

Whether you're exploring startup ecosystems or evaluating investment opportunities, this tool allows you to input historical startup features and instantly predict their outcome (e.g., Acquired, Closed, Operating).

## ✨ Key Features

- **🧠 Dual Model Support**: Employs separate models optimized for both binary and multiclass tasks.
- **🌐 Intuitive Web Interface**: A user-friendly form allowing manual data entry or quick testing.
- **⚡ Smart Prediction Endpoint**: Auto-selects the appropriate ML model based on the specified task type.
- **📂 Batch Processing**: Supports uploading CSV files to perform predictions on multiple startups simultaneously.
- **📊 Interactive & Detailed Results**: View prediction outputs containing both the expected class and confidence probability scores.
- **🛠 RESTful API**: Offers programmatic, JSON-based endpoints for seamless integration into other software.

---

## 🛠️ Technology Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend Web Interface:** HTML, CSS, Jinja2 Templates
- **Deployment & Environments:** Werkzeug, Gunicorn (production), Docker

---

## 🚀 Getting Started

Follow these instructions to set up the project locally on your machine.

### 📋 Prerequisites

- Python 3.8 or higher installed on your system.
- `pip` package manager.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Predictive-Modeling-Startup_Aquisition.git
cd Predictive-Modeling-Startup_Aquisition
```

### 2️⃣ Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Prepare & Train Models

Before running the web app, you need to preprocess your base data (`fe_outcomes2.csv`) and train the predictive models:

```bash
python model_preparation.py
```

*This script will generate `binary_model.pkl`, `multiclass_model.pkl`, and `model_metadata.json` inside the `models/` directory.*

### 4️⃣ Run the Application

Start the local Flask development server:

```bash
python app.py
```

Navigate to `http://localhost:5000` in your web browser to use the interface!

---

## 📡 API Endpoints

You can also use the prediction pipeline programmatically. Here are the core endpoints:

### 👉 `POST /predict` (Smart Endpoint)
Automatically utilizes the correct model based on your `task_type` input (`binary` or `multiclass`).
```json
// Request Body
{
  "task_type": "binary",
  "entity_id": 123,
  "category_code": 42
  // ...other features
}
```

### 👉 `POST /predict-binary`
Strictly predicts using the binary classification model.

### 👉 `POST /predict-multiclass`
Strictly predicts using the multiclass model.

**Sample API Response:**
```json
{
  "prediction": 3,
  "prediction_label": "Status 3",
  "probabilities": {
    "class_1": 0.15,
    "class_2": 0.25,
    "class_3": 0.60
  },
  "confidence": 0.60,
  "model_type": "multiclass"
}
```

### 👉 `GET /api/info`
Retrieves API metadata, schema definitions, and model availability status.

---

## 📁 Project Structure

```text
.
├── app.py                      # Core Flask web server and API routes
├── model_preparation.py        # Script for ML model training and preparation
├── ml_utils.py                 # Custom ML transformers and pipeline components
├── requirements.txt            # Python dependencies
├── fe_outcomes2.csv            # Sourced startup dataset
├── models/                     # Saved model artifacts (generated)
│   ├── binary_model.pkl
│   ├── multiclass_model.pkl
│   └── model_metadata.json
├── templates/                  # Frontend HTML UI pages
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── batch_results.html
│   └── error.html
└── static/                     # Assets (CSS/JS)
```

---

## 🐳 Docker Deployment

To run this application in a Docker container suitable for production:

1. Build the image:
```bash
docker build -t startup-predictor-app .
```
2. Run the container:
```bash
docker run -p 5000:5000 startup-predictor-app
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
<div align="center">
  <i>Developed to make startup acquisition analysis accessible, data-driven, and intuitive.</i>
</div>
