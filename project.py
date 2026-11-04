# ============================================================
#  DATA MINING PROJECT
#  Title      : Student Performance Prediction
#  Domain     : Education
#  Algorithms : Decision Tree | Random Forest | Naive Bayes
#  Dataset    : UCI Student Performance Dataset
#  Link       : https://archive.ics.uci.edu/ml/datasets/Student+Performance
#  Direct CSV : https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv
#
#  Dependencies (install before running):
#      pip install pandas numpy matplotlib seaborn scikit-learn
# ============================================================


# ╔══════════════════════════════════════════════════════════╗
# ║              SECTION 1 : IMPORT LIBRARIES               ║
# ╚══════════════════════════════════════════════════════════╝

import pandas as pd
# pandas  → loads our dataset and lets us work with it like a spreadsheet table

import numpy as np
# numpy   → helps with number arrays and math operations

import matplotlib.pyplot as plt
# matplotlib → the base library for drawing all our charts and graphs

import seaborn as sns
# seaborn  → built on top of matplotlib; makes charts look cleaner and nicer

import warnings
warnings.filterwarnings('ignore')
# hides unnecessary warning messages so output stays clean

# ── The 3 Algorithm Imports ──────────────────────────────────
from sklearn.tree import DecisionTreeClassifier, plot_tree
# DecisionTreeClassifier → the Decision Tree algorithm
# plot_tree              → draws the tree structure as a visual diagram

from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier → builds many decision trees and combines them

from sklearn.naive_bayes import GaussianNB
# GaussianNB → Naive Bayes algorithm, uses probability/chance to predict

# ── Data Preparation Tools ───────────────────────────────────
from sklearn.model_selection import train_test_split
# train_test_split → splits data into 80% training and 20% testing

from sklearn.preprocessing import StandardScaler
# StandardScaler → scales all features to the same range (improves Naive Bayes)

# ── Evaluation / Metric Tools ────────────────────────────────
from sklearn.metrics import (
    accuracy_score,         # what % of all predictions were correct
    precision_score,        # of predicted Pass → how many really passed
    recall_score,           # of actual Pass students → how many did we catch
    f1_score,               # balanced average of precision and recall
    classification_report,  # full table showing all above metrics
    confusion_matrix,       # 2×2 table: correct vs wrong predictions
    roc_curve,              # calculates points to draw the ROC graph
    auc                     # calculates the area under the ROC curve (AUC score)
)

print("=" * 60)
print("   DATA MINING PROJECT : STUDENT PERFORMANCE PREDICTION")
print("=" * 60)
print("✅  All libraries imported successfully\n")


# ╔══════════════════════════════════════════════════════════╗
# ║              SECTION 2 : LOAD THE DATASET               ║
# ╚══════════════════════════════════════════════════════════╝

# The dataset URL — this must also appear in the project report
DATASET_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv"

print("⏳  Loading dataset from URL...")
df = pd.read_csv(DATASET_URL, sep=";")
# pd.read_csv() reads the file from the internet into a dataframe
# sep=";" means the columns in this file are separated by semicolons (;)
# A dataframe is like an Excel table — rows = students, columns = features

print(f"✅  Dataset loaded!  Shape: {df.shape}")
# df.shape → (rows, columns) → should be (395, 33)
# 395 students, 33 columns (features/attributes)

print("\nColumn names in dataset:")
print(list(df.columns))
# prints all 33 column names so we know what data is available


# ╔══════════════════════════════════════════════════════════╗
# ║         SECTION 3 : EXPLORATORY DATA ANALYSIS (EDA)     ║
# ╚══════════════════════════════════════════════════════════╝
# EDA means we explore and understand the data BEFORE applying any algorithm.
# It helps us find patterns, check for problems, and understand relationships.

print("\n" + "─" * 60)
print("  EXPLORATORY DATA ANALYSIS (EDA)")
print("─" * 60)

# ── 3.1 First Look ───────────────────────────────────────────
print("\n📋  First 5 rows of the dataset:")
print(df.head())
# head() shows the first 5 rows — like opening the file and peeking at it

print("\n📋  Last 5 rows of the dataset:")
print(df.tail())
# tail() shows the last 5 rows

# ── 3.2 Basic Statistics ──────────────────────────────────────
print("\n📊  Basic Statistics (numeric columns):")
print(df.describe())
# describe() shows:
# count  = how many values exist
# mean   = average value
# std    = standard deviation (how spread out values are)
# min    = smallest value
# 25%    = 25th percentile
# 50%    = median (middle value)
# 75%    = 75th percentile
# max    = largest value

# ── 3.3 Missing Values ────────────────────────────────────────
print("\n🔍  Missing values per column:")
missing = df.isnull().sum()
# isnull() → marks each cell True if empty, False if filled
# .sum()   → counts True values per column
print(missing[missing > 0] if missing.sum() > 0 else "✅  No missing values found!")
# Good news: this dataset has zero missing values

# ── 3.4 Data Types ────────────────────────────────────────────
print("\n🔍  Data types of each column:")
print(df.dtypes)
# dtypes shows whether each column is int (integer), float (decimal), or object (text)

# ── 3.5 Unique values for key categorical columns ─────────────
print("\n🔍  Unique values in categorical columns:")
for col in ['school', 'sex', 'address', 'Pstatus', 'Mjob', 'Fjob']:
    print(f"  {col}: {df[col].unique()}")
# This shows what categories exist in text-type columns


# ╔══════════════════════════════════════════════════════════╗
# ║              SECTION 4 : CREATE TARGET COLUMN           ║
# ╚══════════════════════════════════════════════════════════╝
# The original dataset has G3 (final grade from 0 to 20).
# We convert it into a binary label: Pass (1) or Fail (0).
# This turns it into a CLASSIFICATION problem.

df['Pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
# lambda is a tiny one-line function
# For each grade value x in the G3 column:
#   if x >= 10 → return 1 (Pass)
#   if x < 10  → return 0 (Fail)

print("\n" + "─" * 60)
print("  TARGET COLUMN CREATED")
print("─" * 60)
print("Pass/Fail distribution:")
print(df['Pass'].value_counts())
print(f"\n  Pass rate: {df['Pass'].mean() * 100:.1f}%")
print(f"  Fail rate: {(1 - df['Pass'].mean()) * 100:.1f}%")


# ╔══════════════════════════════════════════════════════════╗
# ║              SECTION 5 : EDA VISUALIZATIONS             ║
# ╚══════════════════════════════════════════════════════════╝

print("\n📊  Generating EDA charts...")
sns.set_style("whitegrid")
# whitegrid adds light gray gridlines to all charts — looks clean

# ── Chart 1 : Grade Distribution ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df['G3'], bins=20, color='steelblue', kde=True, ax=ax)
# histplot  → draws a histogram (frequency bar chart)
# bins=20   → divides the grade range into 20 groups
# kde=True  → adds a smooth curve showing the overall shape (bell curve)
ax.set_title("Distribution of Final Grades (G3)", fontsize=14, fontweight='bold')
ax.set_xlabel("Final Grade (0–20)")
ax.set_ylabel("Number of Students")
ax.axvline(x=10, color='red', linestyle='--', label='Pass line (10)')
# axvline draws a vertical line at x=10 (the pass/fail boundary)
ax.legend()
plt.tight_layout()
plt.savefig("chart1_grade_distribution.png", dpi=150)
plt.show()
print("  ✅  chart1_grade_distribution.png saved")

# ── Chart 2 : Pass vs Fail Count ──────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x='Pass', data=df, palette=['#e74c3c', '#2ecc71'], ax=ax)
# countplot → bar chart showing count of each category
# palette   → red for Fail (0), green for Pass (1)
ax.set_title("Pass vs Fail Student Count", fontsize=14, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fail (0)', 'Pass (1)'])
ax.set_ylabel("Number of Students")
# Add count labels on top of bars
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("chart2_pass_fail_count.png", dpi=150)
plt.show()
print("  ✅  chart2_pass_fail_count.png saved")

# ── Chart 3 : Study Time vs Grade (Box Plot) ──────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(x='studytime', y='G3', data=df, palette='pastel', ax=ax)
# boxplot shows:
#   box         → middle 50% of grades (25th to 75th percentile)
#   horizontal line inside box → median grade
#   whiskers    → range of typical values
#   dots outside whiskers → outliers (unusual students)
ax.set_title("Study Time vs Final Grade", fontsize=14, fontweight='bold')
ax.set_xlabel("Weekly Study Time  (1=<2hrs  |  2=2-5hrs  |  3=5-10hrs  |  4=>10hrs)")
ax.set_ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("chart3_studytime_vs_grade.png", dpi=150)
plt.show()
print("  ✅  chart3_studytime_vs_grade.png saved")

# ── Chart 4 : Absences vs Grade (Scatter Plot) ────────────────
fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(x='absences', y='G3', data=df,
                hue='Pass', palette={0: '#e74c3c', 1: '#2ecc71'},
                alpha=0.6, ax=ax)
# scatterplot → each dot = one student
# x-axis     → number of absences
# y-axis     → final grade
# hue        → color by Pass/Fail (red=fail, green=pass)
# alpha=0.6  → semi-transparent dots so overlaps are visible
ax.set_title("Absences vs Final Grade", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of Absences")
ax.set_ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("chart4_absences_vs_grade.png", dpi=150)
plt.show()
print("  ✅  chart4_absences_vs_grade.png saved")

# ── Chart 5 : Correlation Heatmap ─────────────────────────────
numeric_cols = ['age', 'studytime', 'failures', 'absences',
                'G1', 'G2', 'G3', 'Medu', 'Fedu',
                'famrel', 'freetime', 'goout', 'Pass']
# Select only numeric columns for correlation

fig, ax = plt.subplots(figsize=(11, 8))
corr_matrix = df[numeric_cols].corr()
# .corr() calculates the Pearson correlation between every pair of columns
# +1.0 → perfect positive relationship (both go up together)
#  0.0 → no relationship
# -1.0 → perfect negative relationship (one goes up, other goes down)

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax,
            annot_kws={"size": 8})
# annot=True    → show the correlation numbers inside each cell
# fmt='.2f'     → show 2 decimal places
# cmap='coolwarm' → blue=negative, white=zero, red=positive
# center=0      → white is at 0 (no relationship)
ax.set_title("Correlation Heatmap of All Numeric Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("chart5_correlation_heatmap.png", dpi=150)
plt.show()
print("  ✅  chart5_correlation_heatmap.png saved")

# ── Chart 6 : Failures vs Grade ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x='failures', y='G3', data=df, palette='Reds', ax=ax)
# shows how students with more past failures tend to get lower grades
ax.set_title("Past Failures vs Final Grade", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of Past Failures (0, 1, 2, 3)")
ax.set_ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("chart6_failures_vs_grade.png", dpi=150)
plt.show()
print("  ✅  chart6_failures_vs_grade.png saved")

print("\n✅  All EDA charts saved!\n")


# ╔══════════════════════════════════════════════════════════╗
# ║          SECTION 6 : DATA PREPROCESSING                 ║
# ╚══════════════════════════════════════════════════════════╝
# Preprocessing = preparing raw data so algorithms can use it properly

print("─" * 60)
print("  DATA PREPROCESSING")
print("─" * 60)

# ── 6.1 Select Features (input columns) ──────────────────────
features = ['age', 'studytime', 'failures', 'absences',
            'G1', 'G2', 'Medu', 'Fedu', 'famrel', 'freetime', 'goout']
# These 11 columns are our INPUT — what the model learns from
# age       → student's age (15–22)
# studytime → weekly study hours: 1=<2h, 2=2-5h, 3=5-10h, 4=>10h
# failures  → number of past class failures (0–3)
# absences  → number of school absences (0–93)
# G1        → first period grade (0–20) — earlier exam score
# G2        → second period grade (0–20) — mid-year exam score
# Medu      → mother's education: 0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher
# Fedu      → father's education (same scale as Medu)
# famrel    → family relationship quality: 1=very bad to 5=excellent
# freetime  → free time after school: 1=very low to 5=very high
# goout     → going out with friends: 1=very low to 5=very high

X = df[features]
# X → input table, shape (395, 11) — 395 students, 11 features

y = df['Pass']
# y → target column, shape (395,) — 1=Pass, 0=Fail for each student

print(f"\n  Features selected : {len(features)} columns")
print(f"  Input  X shape    : {X.shape}")
print(f"  Target y shape    : {y.shape}")
print(f"\n  Feature names: {features}")

# ── 6.2 Train-Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% goes to testing  (≈79 students)
    random_state=42   # fixed random seed so results are reproducible
)
# train_test_split randomly shuffles the 395 students and divides them:
#   80% (≈316 students) → training set  → model learns from this
#   20% (≈79 students)  → testing set   → we test on this (model never saw it)

print(f"\n  Training samples  : {X_train.shape[0]}  (80%)")
print(f"  Testing  samples  : {X_test.shape[0]}   (20%)")

# ── 6.3 Scale Features (for Naive Bayes) ─────────────────────
scaler = StandardScaler()
# StandardScaler transforms each column so mean=0 and std=1
# This ensures no single feature dominates due to large scale differences

X_train_scaled = scaler.fit_transform(X_train)
# fit_transform → learns the mean/std FROM training data, then scales it
# IMPORTANT: we ONLY fit on training data, never test data!

X_test_scaled  = scaler.transform(X_test)
# transform → uses the same mean/std learned from training to scale test data

# Note: Decision Tree and Random Forest don't need scaling
# but Naive Bayes works better with scaled data


# ╔══════════════════════════════════════════════════════════╗
# ║          SECTION 7 : ALGORITHM TRAINING                 ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "─" * 60)
print("  TRAINING ALL 3 MODELS")
print("─" * 60)

# ── Algorithm 1 : Decision Tree ───────────────────────────────
print("\n🌳  Training Decision Tree...")
dt_model = DecisionTreeClassifier(
    max_depth=4,       # tree can only be 4 levels deep
                       # → prevents memorizing training data (overfitting)
    min_samples_split=5,  # a node must have at least 5 samples to be split
    random_state=42    # fixed seed for reproducibility
)
dt_model.fit(X_train, y_train)
# .fit() is the learning step → the model studies X_train and y_train
# It builds a tree by finding the BEST question (feature split) at each node
# Goal: find splits that create the purest groups (all Pass or all Fail)
print("  ✅  Decision Tree trained")
print(f"  Tree depth  : {dt_model.get_depth()}")
print(f"  Tree leaves : {dt_model.get_n_leaves()}")

# ── Algorithm 2 : Random Forest ──────────────────────────────
print("\n🌲  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,  # build 100 decision trees
    max_depth=6,       # each tree limited to 6 levels
    random_state=42
)
rf_model.fit(X_train, y_train)
# .fit() builds 100 decision trees, each on a RANDOM subset of:
#   - rows (bootstrap sampling → random selection of students)
#   - features (random subset of features at each split)
# Final prediction = majority vote of all 100 trees
# More trees = more stable, less overfitting
print("  ✅  Random Forest trained")
print(f"  Number of trees : {rf_model.n_estimators}")

# ── Algorithm 3 : Naive Bayes ─────────────────────────────────
print("\n🔵  Training Naive Bayes...")
nb_model = GaussianNB()
# GaussianNB = Gaussian Naive Bayes
# "Gaussian"  → assumes features follow a bell-curve (normal distribution)
# "Naive"     → assumes all features are INDEPENDENT of each other
# It works by calculating:
#   P(Pass | features) → probability of passing given the feature values
#   P(Fail | features) → probability of failing given the feature values
# Whichever probability is higher → that's the prediction

nb_model.fit(X_train_scaled, y_train)
# We use SCALED features for Naive Bayes (better performance)
print("  ✅  Naive Bayes trained")

print("\n✅  All 3 models trained successfully!")


# ╔══════════════════════════════════════════════════════════╗
# ║              SECTION 8 : PREDICTIONS                    ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "─" * 60)
print("  GENERATING PREDICTIONS")
print("─" * 60)

# ── Class label predictions (0 or 1) ─────────────────────────
dt_pred = dt_model.predict(X_test)
# .predict() → model gives a final answer: 0 (Fail) or 1 (Pass)
# for each of the 79 test students

rf_pred = rf_model.predict(X_test)
# same for Random Forest

nb_pred = nb_model.predict(X_test_scaled)
# Naive Bayes uses SCALED test data

# ── Probability predictions (for ROC curve) ──────────────────
dt_prob = dt_model.predict_proba(X_test)[:, 1]
# .predict_proba() → instead of 0/1, gives a probability between 0.0 and 1.0
# [:,1] → we only want the probability of class 1 (Pass)
# Example: 0.85 means "85% chance this student will pass"
# We need these probabilities to draw the ROC curve

rf_prob = rf_model.predict_proba(X_test)[:, 1]

nb_prob = nb_model.predict_proba(X_test_scaled)[:, 1]

print("\n  Sample predictions (first 10 test students):")
print(f"  Actual        : {list(y_test[:10])}")
print(f"  Decision Tree : {list(dt_pred[:10])}")
print(f"  Random Forest : {list(rf_pred[:10])}")
print(f"  Naive Bayes   : {list(nb_pred[:10])}")


# ╔══════════════════════════════════════════════════════════╗
# ║           SECTION 9 : MODEL EVALUATION                  ║
# ╚══════════════════════════════════════════════════════════╝

def evaluate_model(model_name, y_true, y_pred, y_prob):
    """
    Calculates all evaluation metrics for one model.

    Parameters:
      model_name  : name string (just for display)
      y_true      : actual correct labels (ground truth)
      y_pred      : model's predicted labels
      y_prob      : model's predicted probabilities (for ROC/AUC)

    Returns:
      Dictionary containing all metric scores and ROC data
    """

    # ── Accuracy ───────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    # Formula: (True Positives + True Negatives) / Total Predictions
    # Example: 75 correct out of 79 → accuracy = 75/79 = 0.949 = 94.9%

    # ── Precision ──────────────────────────────────────────────
    prec = precision_score(y_true, y_pred, zero_division=0)
    # Formula: True Positives / (True Positives + False Positives)
    # "Of all students we PREDICTED as Pass, how many ACTUALLY passed?"
    # High precision → few false alarms (we're confident when we say Pass)

    # ── Recall (Sensitivity) ───────────────────────────────────
    rec = recall_score(y_true, y_pred, zero_division=0)
    # Formula: True Positives / (True Positives + False Negatives)
    # "Of all students who ACTUALLY passed, how many did we CORRECTLY predict?"
    # High recall → we miss very few actual passing students

    # ── F1-Score ───────────────────────────────────────────────
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Formula: 2 × (Precision × Recall) / (Precision + Recall)
    # F1 is the HARMONIC MEAN of precision and recall
    # Useful when both precision and recall matter equally
    # It's always between 0 and 1 (higher = better)

    # ── ROC Curve Data ─────────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # roc_curve tests ALL possible probability thresholds
    # For each threshold it calculates:
    #   fpr = False Positive Rate = FP / (FP + TN)
    #         "How often do we wrongly predict Pass when student actually Fails?"
    #   tpr = True Positive Rate  = TP / (TP + FN)
    #         "How often do we correctly predict Pass when student actually Passes?"
    # These points form the ROC curve when plotted

    # ── AUC Score ──────────────────────────────────────────────
    roc_auc = auc(fpr, tpr)
    # auc = Area Under the ROC Curve
    # 1.00 → perfect model
    # 0.90 → excellent
    # 0.80 → good
    # 0.70 → fair
    # 0.50 → useless (same as random guessing)

    # ── Print Results ──────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print(f"  MODEL : {model_name}")
    print(f"{'═' * 50}")
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1-Score  : {f1   * 100:.2f}%")
    print(f"  AUC Score : {roc_auc:.4f}")
    print(f"\n  Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fail', 'Pass']))
    # classification_report shows precision, recall, f1 SEPARATELY for
    # both Fail class and Pass class, plus weighted averages

    return {
        'name'   : model_name,
        'acc'    : acc,
        'prec'   : prec,
        'rec'    : rec,
        'f1'     : f1,
        'auc'    : roc_auc,
        'fpr'    : fpr,
        'tpr'    : tpr,
        'y_pred' : y_pred,
        'y_prob' : y_prob
    }

# ── Run evaluation for all 3 models ───────────────────────────
print("\n" + "─" * 60)
print("  MODEL EVALUATION RESULTS")
print("─" * 60)

dt_results = evaluate_model("Decision Tree", y_test, dt_pred, dt_prob)
rf_results = evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
nb_results = evaluate_model("Naive Bayes",   y_test, nb_pred, nb_prob)

all_results = [dt_results, rf_results, nb_results]


# ╔══════════════════════════════════════════════════════════╗
# ║          SECTION 10 : RESULT VISUALIZATIONS             ║
# ╚══════════════════════════════════════════════════════════╝

print("\n📊  Generating result charts...")

# ── Chart 7 : Confusion Matrices (all 3 side by side) ─────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# subplots(1,3) → 1 row, 3 columns → 3 charts side by side

for idx, result in enumerate(all_results):
    cm = confusion_matrix(y_test, result['y_pred'])
    # Confusion Matrix layout:
    #                   Predicted Fail  |  Predicted Pass
    # Actual Fail  →  [ True Negatives  |  False Positives ]
    # Actual Pass  →  [ False Negatives |  True Positives  ]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'],
                yticklabels=['Fail', 'Pass'],
                ax=axes[idx], cbar=False)
    axes[idx].set_title(f"{result['name']}\nAcc: {result['acc']*100:.1f}%",
                        fontweight='bold')
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("Actual Label")

fig.suptitle("Confusion Matrices – All 3 Models", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("chart7_confusion_matrices.png", dpi=150)
plt.show()
print("  ✅  chart7_confusion_matrices.png saved")

# ── Chart 8 : ROC Curves (all 3 on one graph) ─────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Draw the diagonal reference line (random guessing baseline)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guess (AUC = 0.50)')
# A GOOD model should curve towards the TOP-LEFT corner
# The dashed diagonal = useless model (50% chance, like coin flip)

colors = ['#2980b9', '#27ae60', '#e74c3c']
# blue for Decision Tree, green for Random Forest, red for Naive Bayes

for result, color in zip(all_results, colors):
    ax.plot(result['fpr'], result['tpr'],
            label=f"{result['name']}  (AUC = {result['auc']:.3f})",
            color=color, linewidth=2.5)
    # fpr on x-axis = False Positive Rate (how often we cry wolf)
    # tpr on y-axis = True Positive Rate (how often we find real cases)
    # AUC number in legend = area under this curve

ax.fill_between(rf_results['fpr'], rf_results['tpr'],
                alpha=0.08, color='#27ae60')
# Shades the area under the best model's curve (visual aid)

ax.set_title("ROC Curves – Model Comparison", fontsize=14, fontweight='bold')
ax.set_xlabel("False Positive Rate (FPR)\n← We want this LOW (left side)")
ax.set_ylabel("True Positive Rate (TPR)\n← We want this HIGH (top side)")
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
plt.tight_layout()
plt.savefig("chart8_roc_curves.png", dpi=150)
plt.show()
print("  ✅  chart8_roc_curves.png saved")

# ── Chart 9 : Side-by-Side Metric Bar Chart ───────────────────
metrics_list  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
metric_keys   = ['acc', 'prec', 'rec', 'f1', 'auc']
model_names   = [r['name'] for r in all_results]
model_colors  = ['#2980b9', '#27ae60', '#e74c3c']

x_pos  = np.arange(len(metrics_list))   # [0, 1, 2, 3, 4]
width  = 0.25                            # width of each bar

fig, ax = plt.subplots(figsize=(13, 6))

for i, (result, color) in enumerate(zip(all_results, model_colors)):
    values = [result[k] for k in metric_keys]
    # Extract the 5 metric values for this model

    offset = (i - 1) * width
    # i=0 → offset=-0.25 (left bar)
    # i=1 → offset= 0.00 (center bar)
    # i=2 → offset=+0.25 (right bar)

    bars = ax.bar(x_pos + offset, values, width,
                  label=result['name'], color=color, alpha=0.85,
                  edgecolor='white', linewidth=0.8)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_title("Model Comparison – All Metrics", fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_list, fontsize=12)
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score (0.0 → 1.0 scale)")
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("chart9_model_comparison_bar.png", dpi=150)
plt.show()
print("  ✅  chart9_model_comparison_bar.png saved")

# ── Chart 10 : Decision Tree Visual ───────────────────────────
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(
    dt_model,
    feature_names=features,            # label each node with feature name
    class_names=['Fail', 'Pass'],       # label leaf nodes with class name
    filled=True,                        # color nodes by majority class
    rounded=True,                       # rounded box corners (looks nicer)
    fontsize=9,
    ax=ax
)
ax.set_title("Decision Tree Structure (max_depth=4)\n"
             "How to read: Left branch = condition TRUE, Right = FALSE",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("chart10_decision_tree.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅  chart10_decision_tree.png saved")

# ── Chart 11 : Feature Importance (Random Forest) ─────────────
importances = rf_model.feature_importances_
# .feature_importances_ is calculated automatically after training
# It measures how often and how much each feature reduces impurity
# Higher score = that feature was more useful for making predictions

importance_df = pd.DataFrame({
    'Feature'   : features,
    'Importance': importances
}).sort_values('Importance', ascending=False)
# sort highest to lowest so most important features appear at top

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
               color=plt.cm.viridis(np.linspace(0.2, 0.8, len(features))))
# barh → horizontal bar chart (features on y-axis, importance on x-axis)
# plt.cm.viridis → gradient color scheme from dark purple to yellow-green

for bar, val in zip(bars, importance_df['Importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9)

ax.set_title("Feature Importance – Random Forest\n"
             "(Which features matter most for predicting Pass/Fail?)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature Name")
plt.tight_layout()
plt.savefig("chart11_feature_importance.png", dpi=150)
plt.show()
print("  ✅  chart11_feature_importance.png saved")

print("\n✅  All result charts saved!")


# ╔══════════════════════════════════════════════════════════╗
# ║          SECTION 11 : FINAL SUMMARY TABLE               ║
# ╚══════════════════════════════════════════════════════════╝

summary_df = pd.DataFrame({
    'Model'     : [r['name']       for r in all_results],
    'Accuracy'  : [f"{r['acc']*100:.2f}%"  for r in all_results],
    'Precision' : [f"{r['prec']*100:.2f}%" for r in all_results],
    'Recall'    : [f"{r['rec']*100:.2f}%"  for r in all_results],
    'F1-Score'  : [f"{r['f1']*100:.2f}%"   for r in all_results],
    'AUC Score' : [f"{r['auc']:.4f}"       for r in all_results],
})

print("\n" + "═" * 70)
print("                    FINAL SUMMARY TABLE")
print("═" * 70)
print(summary_df.to_string(index=False))
print("═" * 70)

# Identify best model
best_by_auc = max(all_results, key=lambda r: r['auc'])
best_by_acc = max(all_results, key=lambda r: r['acc'])
best_by_f1  = max(all_results, key=lambda r: r['f1'])

print(f"\n  🏆  Best by AUC      : {best_by_auc['name']}  ({best_by_auc['auc']:.4f})")
print(f"  🏆  Best by Accuracy : {best_by_acc['name']}  ({best_by_acc['acc']*100:.2f}%)")
print(f"  🏆  Best by F1-Score : {best_by_f1['name']}   ({best_by_f1['f1']*100:.2f}%)")

print("\n" + "═" * 70)
print("  INTERPRETATION GUIDE")
print("─" * 70)
print("""
  Metric      What it means in plain language
  ─────────────────────────────────────────────────────────────────
  Accuracy  → Out of ALL 79 test students, what % did we predict correctly?
  Precision → Of all students we called 'Pass', what % actually passed?
              High precision = fewer false alarms.
  Recall    → Of all students who actually passed, what % did we catch?
              High recall = we miss very few actual passers.
  F1-Score  → A single balanced score combining Precision + Recall.
              Use this when both false alarms AND misses matter.
  AUC       → How well the model separates Pass from Fail overall.
              1.00 = perfect | 0.90 = excellent | 0.50 = useless
  ─────────────────────────────────────────────────────────────────
  Why Random Forest usually wins:
  → It builds 100 trees instead of just 1, so it's less likely to
    make mistakes. More diverse opinions → better final vote.
""")

print("═" * 70)
print("  ✅  PROJECT COMPLETE!  All 11 charts saved.")
print("═" * 70)
