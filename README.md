# 🚢 Titanic Survival Prediction

An end-to-end Machine Learning project that predicts whether a passenger survived the Titanic disaster using classification algorithms.

This project follows a complete Data Science workflow, starting from data exploration and preprocessing to model training, evaluation, and deployment through a Streamlit web application.

---

# 📌 Project Overview

The goal of this project is to build a Machine Learning model that predicts passenger survival based on demographic and travel-related information.

The model uses important passenger features such as:

- Passenger Class
- Gender
- Age
- Fare
- Family Information
- Embarked Port
- Passenger Title

The complete Machine Learning pipeline includes:

- Data Collection
- Data Understanding
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation
- Model Comparison
- Model Saving
- Streamlit Deployment

---

# 🚀 Application Features

The project includes an interactive Streamlit web application that allows users to:

- Enter passenger details
- Predict survival chances
- View survival probability
- View death probability
- Display passenger information summary

---

# 📂 Project Structure

```
Titanic-Survival-Prediction/
│
├── data/
|   ├── test.csv
|   ├── titanic_cleaned.csv
│   └── train.csv
│
├── notebook/
│   └── Titanic_Survival_Prediction.ipynb
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Exploratory Data Analysis (EDA)

A detailed Exploratory Data Analysis was performed to understand the dataset and identify important patterns.

EDA included:

- Dataset overview
- Data type analysis
- Missing value detection
- Duplicate value checking
- Statistical summary
- Target variable distribution
- Numerical feature analysis
- Categorical feature analysis
- Correlation analysis
- Outlier detection

---

# ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

- Handling missing values
- Removing unnecessary columns
- Feature selection
- Categorical encoding
- Feature scaling
- Train-test split

### Feature Engineering

New features were created to improve model performance:

- Family Size
- Is Alone
- Age Group
- Passenger Title

---

# 🤖 Machine Learning Models

Multiple classification algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- XGBoost Classifier

---

# 📈 Model Performance

| Model | Accuracy |
|------|---------:|
| **XGBoost** | **82.12%** |
| Support Vector Machine | 81.56% |
| Random Forest | 81.56% |
| Logistic Regression | 81.01% |
| Decision Tree | 79.89% |
| K-Nearest Neighbors | 78.21% |
| Gaussian Naive Bayes | 78.21% |

---

# 🏆 Best Performing Model

The best performing model was:

## XGBoost Classifier

The XGBoost Classifier achieved the highest test accuracy (**82.12%**) among all evaluated models. It was selected as the final model for deployment because of its strong predictive performance and good generalization on unseen data.
---

# 🛠 Technologies Used

## Programming Language

- Python

## Data Analysis & Processing

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn
- XGBoost
- Joblib

## Deployment

- Streamlit

## Development Tools

- Jupyter Notebook
- VS Code

---

# ▶️ How to Run the Project Locally

## 1. Clone the Repository

```bash
git clone <repository-url>
```

## 2. Navigate to Project Directory

```bash
cd Titanic-Survival-Prediction
```

## 3. Create Virtual Environment

```bash
python -m venv venv
```

## 4. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

## 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

## 6. Run Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser:

```
http://localhost:8501
```

---

# 📦 Saved Model Files

The trained Machine Learning model and preprocessing scaler are saved using Joblib.

```
models/
│
├── model.pkl
└── scaler.pkl
```

These files are used by the Streamlit application for real-time prediction.

---

# 🌐 Deployment

The application is prepared for deployment using:

- Hugging Face Spaces
- Streamlit Cloud

The deployed application provides an interactive interface for survival prediction.

---

# 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Complete Machine Learning workflow
- Exploratory Data Analysis
- Data preprocessing techniques
- Feature engineering
- Classification algorithms
- Model evaluation
- Model comparison
- Model deployment
- Building ML applications using Streamlit

---

# 🔮 Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Explainable AI using SHAP
- Docker containerization
- Cloud deployment

---

# Author

**Alisha Sajjad**

Machine Learning | Python | Data Science

---

⭐ If you found this project useful, consider giving it a star on GitHub.