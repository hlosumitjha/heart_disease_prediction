# ❤️ Heart Disease Prediction App (Logistic Regression + Streamlit)

✅ **Live Demo:**  
👉 https://hlosumitjha-heart-disease-prediction-appapp-3qdoma.streamlit.app/

A clean and modern ML-powered app to predict the probability of heart disease based on patient health data.

---

## ✅ Features

### 🔹 Machine Learning
- Logistic Regression classifier  
- Preprocessing (OneHotEncoder + StandardScaler)  
- Supports probability prediction  
- Feature engineering included  
- High test accuracy  

### 🔹 Application Features
- Clean medical-themed UI (red-white theme)  
- Categorical dropdowns  
- Numeric steppers (+ / −)  
- Real-time single predictions  
- Batch prediction via CSV  
- Downloadable results  
- Probability visualization  

---

## 📁 Project Structure

```
heart_disease_prediction/
│
├── app/
│   └── app.py
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── data/
│   └── HeartDiseaseTrain-Test.csv
│
├── model/
│   └── heart_model.pkl
│
├── README.md
└── requirements.txt
```


---

## ✅ Local Installation Guide

### 1️⃣ Create virtual environment  
```bash
python -m venv venv
Activate the environment:

Windows

bash
Copy code
venv\Scripts\activate
Mac/Linux

bash
Copy code
source venv/bin/activate
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Train the model
(Optional if heart_model.pkl already exists)

bash
Copy code
python src/train_model.py
This trains the Logistic Regression model and saves it inside the model/ folder.

4️⃣ Run the Streamlit app
bash
Copy code
streamlit run app/app.py
Your browser will open automatically:

👉 http://localhost:8501

✅ Deployment (Streamlit Cloud)
Your deployed app is live here:
✅ https://hlosumitjha-heart-disease-prediction-appapp-3qdoma.streamlit.app/

Deployment Steps:
Push the entire project to GitHub

Open https://share.streamlit.io

Connect your GitHub repo

Set the entry point:

bash
Copy code
app/app.py
Select Python version

Add requirements.txt for dependencies

Deploy ✅

⚙️ Requirements
nginx
Copy code
streamlit
pandas
matplotlib
scikit-learn
numpy
Install using:

bash
Copy code
pip install -r requirements.txt
❗ Troubleshooting
🔴 UI not updating
Clear cache

Press Ctrl + F5

🔴 Model not loading
Run training again:

bash
Copy code
python src/train_model.py
🔴 Wrong CSV format
Ensure CSV has exact same columns as training data.

🚀 Future Enhancements
Add SHAP explainability

Add dark/light theme switch

Add database for saving results

Add user login

Add more ML models (Random Forest, XGBoost)

🧑‍💻 Author
Sumit Kumar Jha
Full Stack Developer & ML Enthusiast

yaml
Copy code
