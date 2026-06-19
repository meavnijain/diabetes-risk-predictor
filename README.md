# 🏥 AI Diabetes Risk Predictor

A diagnostic analytics web application designed to evaluate patient biometric parameters and predict diabetes health risks. The system utilizes an ensemble machine learning architecture combined with an interactive user interface to deliver real-time risk calculations.

🚀 **[Live Demo Link](https://diabetes-risk-predictor-byavnijain.streamlit.app/)**

---

## 📊 How It Works (Machine Learning Pipeline)

The application handles clinical prediction tasks through structured tabular data analysis:

1. **Feature Engineering:**
   The model processes four critical biological indicators inputted via the client dashboard:
   - **Glucose Level (mg/dL):** Blood sugar concentration measurements.
   - **Blood Pressure (mm Hg):** Diastolic pressure metrics.
   - **Body Mass Index (BMI):** Weight-to-height relative scale tracking corporate obesity data.
   - **Age (Years):** Chronological patient risk indexing.

2. **Ensemble Classification Architecture (`Random Forest Classifier`):**
   Instead of relying on a single decision tree, this pipeline utilizes a **Random Forest Ensemble**. It constructs multiple independent decision trees during training and merges their predictions (voting matrix) to eliminate individual tree bias, resulting in highly accurate risk evaluation probabilities (`predict_proba`).

---

## 🛠️ Tech Stack & Architecture

- **Programming Language:** Python 3
- **Machine Learning & Modeling:** Scikit-Learn, Pandas, NumPy
- **Application Framework & Core UI:** Streamlit Engine
- **Cloud Infrastructure & Hosting:** Render Cloud Platform / Streamlit Cloud

---

## ⚙️ Local Installation & Execution

Follow these environment configurations to execute the platform on a local architecture:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/meavnijain/diabetes-risk-predictor
   cd diabetes-risk-predictor
   ```

2. **Establish environment requirements:**
   Ensure you have Python installed, then build dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Boot up the local system server:**
   ```bash
   streamlit run app.py
   ```

---

## 📈 Planned Structural Enhancements
- Transition system architecture from a static internal matrix to the **PIMA Indians Diabetes Database**.
- Integrate data scaling workflows using standard Z-score statistical scaling (`StandardScaler`) to balance input variances.
- Deploy an optimization matrix utilizing hyperparameter tuning algorithms (`GridSearchCV`) to maximize structural model precision.
