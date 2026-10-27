# 📉 Customer Churn Prediction: SVM vs KNN

A machine learning project that predicts whether a bank customer will leave (churn), comparing two classification models: **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)**.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Business Implications](#-business-implications)
- [Future Improvements](#-future-improvements)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Repository Structure](#-repository-structure)
- [Author](#-author)

---

## 🎯 Project Overview

Customer churn is expensive: keeping an existing customer is usually cheaper than acquiring a new one. This project builds and evaluates two classifiers that predict which customers are likely to leave, and examines which model is better suited to different business goals.

**Objectives**

- Explore and preprocess the churn dataset
- Train an SVM (RBF kernel) and a KNN model
- Select the optimal `K` for KNN using the elbow method
- Compare models using accuracy, precision, recall, and F1-score
- Translate the results into business recommendations

---

## 📊 Dataset

- **File:** `Churn_Modelling.csv` (bank customer churn dataset)
- **Target variable:** `Exited` (1 = customer churned, 0 = customer stayed)
- **Features include:** credit score, geography, gender, age, tenure, balance, number of products, credit card status, active membership, and estimated salary

> The dataset is **imbalanced**: far fewer customers churn than stay, which strongly affects model recall (see results below).

---

## 🔄 Project Workflow

1. **Data loading and inspection:** shape, data types, missing values
2. **Target analysis:** churn class distribution
3. **Data cleaning:** dropped non-predictive columns (`RowNumber`, `CustomerId`, `Surname`)
4. **Encoding:**
   - One-hot encoding for `Geography` (with `drop_first=True` to avoid multicollinearity)
   - Label encoding for `Gender`
5. **Train/test split:** 80/20, stratified on the target, `random_state=42`
6. **Feature scaling:** `StandardScaler` (fit on training data only to avoid data leakage)
7. **Modeling:**
   - SVM with the default RBF kernel
   - KNN with `K` chosen via the elbow method (K = 1 to 20), optimal **K = 5**
8. **Evaluation:** confusion matrix, accuracy, and classification report

---

## 🏆 Results

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|-------|:--------:|:-----------------:|:--------------:|:----------------:|
| **SVM (RBF)** | **0.8610** | **0.83** | **0.40** | **0.54** |
| KNN (K=5) | 0.8240 | 0.62 | 0.34 | 0.44 |

✅ **SVM outperforms KNN** on accuracy, precision, and F1-score for this dataset and split.

---

## 💡 Key Insights

- **SVM is more reliable when it flags a churner.** It is correct 83% of the time, versus 62% for KNN.
- **Both models have low recall.** They miss roughly 60% (SVM) and 66% (KNN) of customers who actually churned.
- **Accuracy alone is misleading.** A model can look strong overall while still missing most of the minority class.
- **F1-score favors SVM** (0.54 vs 0.44), making it the better overall model for the churn class.

---

## 💼 Business Implications

| Business Goal | Recommended Approach |
|---------------|----------------------|
| Spend retention budget efficiently on customers likely to leave | **SVM** (high precision) |
| Catch as many potential churners as possible | Neither model is sufficient yet; address class imbalance first |

---

## 🚀 Future Improvements

- [ ] **Handle class imbalance** with SMOTE, oversampling, or `class_weight='balanced'`
- [ ] **Hyperparameter tuning** with GridSearchCV (`C` and `gamma` for SVM; distance metrics and weights for KNN)
- [ ] **Feature engineering**, for example balance-to-salary ratio and product utilization
- [ ] **Try additional models** such as Random Forest, XGBoost, and Logistic Regression
- [ ] **Optimize the decision threshold** or use precision-recall curves rather than accuracy alone

---

## 🛠 Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn (SVC, KNeighborsClassifier, StandardScaler, metrics)
- **Environment:** Jupyter Notebook / Google Colab

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Add the dataset**

   Place `Churn_Modelling.csv` in the project folder. The notebook currently reads from `/content/Churn_Modelling.csv` (Google Colab path), so update this line to match your setup:
   ```python
   df = pd.read_csv('Churn_Modelling.csv')
   ```

4. **Launch the notebook**
   ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```

---

## 📁 Repository Structure

```
├── Customer_Churn_Prediction.ipynb   # Full analysis and modeling notebook
├── Customer_Churn_Prediction.pdf     # Written report of the results
├── Churn_Modelling.csv               # Dataset (add this yourself)
└── README.md
```

---

## 👤 Author

**Your Name**

- LinkedIn: [your-linkedin-url](https://www.linkedin.com/in/your-profile)
- GitHub: [@your-username](https://github.com/your-username)

---

⭐ If you found this project useful, consider giving it a star!
