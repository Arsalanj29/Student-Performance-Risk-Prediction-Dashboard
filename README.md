# 🎓 EduRisk — Student Performance & Risk Prediction Dashboard

> A full-stack ML-powered analytics dashboard that predicts at-risk students using attendance, study habits, health scores, and exam performance.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🌐 Live Demo

View: https://student-performance-risk-prediction-dashboard-ddwgejacjypcbnay.streamlit.app/

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Overview Dashboard** | KPI cards, risk donut chart, score distributions, auto-generated insights |
| 📈 **Deep Analysis** | Correlation heatmap, demographic breakdowns, lifestyle factors, violin plots |
| 🤖 **ML Model** | Random Forest, AUC-ROC, confusion matrix, feature importance ranking |
| 🔮 **Predict Student** | Real-time risk scoring for any student profile with personalized recommendations |
| 🎛️ **Sidebar Filters** | Filter by grade, risk level, school type; adjust dataset size |
| 📥 **Data Export** | Download the full synthetic dataset as CSV |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (custom CSS, Google Fonts, animated UI)
- **Data**: Synthetic student dataset (generated in-app with NumPy/Pandas)
- **ML Model**: Random Forest Classifier (Scikit-learn)
  - 200 trees · Max depth 8 · Class-balanced weights · 5-fold cross-validation
- **Visualization**: Matplotlib + Seaborn (custom styled charts)
- **Deployment**: Streamlit Community Cloud (free, no server needed)

---

## 🚀 Deploy in 3 Steps (Streamlit Cloud — Free)

### Step 1 — Fork / Push to GitHub

```bash
# Clone or create a new GitHub repo, then add these files:
# app.py
# requirements.txt
# README.md
```

Make sure your repo contains:
```
📁 your-repo/
├── app.py
├── requirements.txt
└── README.md
```

### Step 2 — Connect to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Select your repo, branch (`main`), and set the main file to `app.py`
4. Click **"Deploy!"**

### Step 3 — Share Your Link

Streamlit will give you a URL like:
```
https://your-app-name.streamlit.app
```

That's it — **one link, fully live, shareable worldwide. 🌍**

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/edurisk-dashboard.git
cd edurisk-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
edurisk-dashboard/
│
├── app.py              # 🔥 Main application (all-in-one)
├── requirements.txt    # 📦 Python dependencies
└── README.md           # 📖 This file
```

> **No external data files needed** — all student data is synthetically generated inside the app.

---

## 🧠 Machine Learning Pipeline

```
Raw Features (8)
  ↓
StandardScaler (normalization)
  ↓
RandomForestClassifier
  ├── 200 estimators
  ├── max_depth = 8
  ├── class_weight = 'balanced'
  └── cross_val (5-fold)
  ↓
Predictions → Risk Score → Recommendations
```

**Features used:**
- Study Hours per day
- Attendance percentage
- Previous term grade
- Family Support (binary)
- Internet Access (binary)
- Extra Curricular (binary)
- Health Score (1–5)
- Number of Absences

**Target:** `At_Risk` → 1 if Exam Score < 50, else 0

---

## 🎨 Design Highlights

- **Color palette**: Soft green, sky blue, light yellow on a mint-tinted background
- **Typography**: Playfair Display (headings) + Space Grotesk (body) + JetBrains Mono
- **Charts**: Custom-styled Matplotlib/Seaborn (no default themes)
- **Custom CSS**: Hero bar, KPI grid, risk badges, insight pills, progress bars

---

## 📸 Screenshots

| Page | Preview |
|---|---|
| Overview | KPI cards + donut chart + score distribution |
| Deep Analysis | Correlation heatmap + violin plots |
| ML Model | Confusion matrix + feature importance |
| Predict | Real-time risk gauge + recommendations |

---

## 👨‍💻 Author

Built as a **portfolio project** for fresh graduates in Data Science / ML.  
Feel free to fork, customize, and deploy as your own! 🎓

---

## 📄 License

MIT License — free to use, modify, and distribute.
