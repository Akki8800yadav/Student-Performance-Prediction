# 🎓 Student Performance Prediction System

An end-to-end Data Mining and Machine Learning project that predicts student academic performance using multiple classification algorithms and interactive visualizations.

This project analyzes student-related academic, social, and behavioral factors to predict whether a student will pass or fail.

---

# 📌 Project Objectives

* Predict student pass/fail outcomes
* Compare multiple machine learning algorithms
* Perform exploratory data analysis (EDA)
* Visualize performance metrics and feature importance
* Build an interactive Streamlit dashboard

---

# 🛠 Technologies Used

## Programming

* Python

## Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* streamlit

## Machine Learning Algorithms

* Decision Tree
* Random Forest
* Naive Bayes

---

# 📊 Dataset Information

Dataset: UCI Student Performance Dataset

Source:
https://archive.ics.uci.edu/ml/datasets/Student+Performance

Dataset contains:

* 395 student records
* 33 attributes/features

Example Features:

* Age
* Study Time
* Absences
* Previous Grades (G1, G2)
* Family Relationship
* Parent Education
* Failures

Target Variable:

* Pass (1)
* Fail (0)

The target column is generated from final grade (G3):

* G3 ≥ 10 → Pass
* G3 < 10 → Fail

---

# 📈 Exploratory Data Analysis (EDA)

EDA performed in this project includes:

* Grade distribution analysis
* Pass vs fail comparison
* Study time vs grades
* Absences vs performance
* Correlation heatmaps
* Past failures analysis
* Feature importance visualization

Key insights discovered:

* Previous grades strongly influence final performance
* High absences reduce student performance
* More study time generally improves grades
* Past failures negatively affect results

---

# 🤖 Machine Learning Models

## 1. Decision Tree

A tree-based algorithm that splits data using conditions and decision rules.

## 2. Random Forest

An ensemble learning algorithm using multiple decision trees for better accuracy and reduced overfitting.

## 3. Naive Bayes

A probability-based classification algorithm using Bayes theorem.

---

# 📏 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

---

# 🏆 Model Comparison

All three algorithms are trained and compared using the same dataset and evaluation metrics.

The project includes:

* ROC curve comparison
* Side-by-side metric visualization
* Confusion matrices
* Feature importance analysis

---

# 🌐 Streamlit Dashboard

This project includes an interactive Streamlit web application for:

* Dataset exploration
* EDA visualization
* Model comparison
* Performance metrics
* Interactive analysis

Run the app using:

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
Student-Performance-Prediction-System/
│
├── app.py
├── project.py
├── final_prediction_of_students.ipynb
├── student_performance_data.csv
├── requirements.txt
├── README.md
│
├── charts/
│   ├── chart1_grade_distribution.png
│   ├── chart2_pass_fail_count.png
│   ├── chart3_studytime_vs_grade.png
│   ├── chart4_absences_vs_grade.png
│   ├── chart5_correlation_heatmap.png
│   ├── chart6_failures_vs_grade.png
│   ├── chart7_confusion_matrices.png
│   ├── chart8_roc_curves.png
│   ├── chart9_model_comparison_bar.png
│   ├── chart10_decision_tree.png
│   └── chart11_feature_importance.png
```

---

# ▶️ How to Run the Project

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/Student-Performance-Prediction-System.git
```

## 2. Move into Project Folder

```bash
cd Student-Performance-Prediction-System
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

* final_prediction_of_students.ipynb

## 5. Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 📊 Results

* Random Forest achieved the best overall performance
* Previous grades (G1 and G2) were the most important features
* ROC-AUC comparison showed strong classification performance
* Feature importance analysis identified major academic success factors

---

# 💡 Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Classification Algorithms
* Model Evaluation
* ROC-AUC Analysis
* Feature Engineering
* Streamlit Dashboard Development
* Machine Learning Workflow

---

# 🚀 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Deployment on cloud platforms
* Real-time prediction system
* SHAP explainability integration
* Advanced ensemble techniques

---

# 👩‍💻 Author

Aishwarya Reddy

Data Science & Machine Learning Enthusiast

-
