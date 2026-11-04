# ============================================================
#  DATA MINING PROJECT – STREAMLIT WEB APPLICATION
#  Run this app with:   streamlit run app.py
#
#  Install dependencies first:
#      pip install streamlit pandas numpy matplotlib seaborn scikit-learn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance - Data Mining",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #3a3d5c;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin: 6px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-weight: bold; margin: 6px 0; }
    .metric-value.green  { color: #2ecc71; }
    .metric-value.blue   { color: #3498db; }
    .metric-value.orange { color: #f39c12; }
    .metric-value.red    { color: #e74c3c; }
    .metric-value.purple { color: #9b59b6; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a1d2e, transparent);
        border-left: 4px solid #3498db;
        padding: 10px 16px;
        margin: 20px 0 10px 0;
        border-radius: 0 8px 8px 0;
        font-size: 20px;
        font-weight: bold;
        letter-spacing: 0.5px;
    }

    /* Algorithm explanation cards */
    .algo-card {
        background: #1a1d2e;
        border: 1px solid #3a3d5c;
        border-radius: 10px;
        padding: 20px;
        margin: 8px 0;
    }
    .algo-title { font-size: 17px; font-weight: bold; margin-bottom: 8px; }
    .algo-body  { font-size: 14px; color: #aaa; line-height: 1.7; }

    /* Winner banner */
    .winner-banner {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 16px 0;
    }

    /* Info boxes */
    .info-box {
        background: #1a2744;
        border: 1px solid #2980b9;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #0a0d1a; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #aaa; font-size: 15px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #3498db; font-weight: bold; }

    /* Hide streamlit branding */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD & CACHE DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load dataset from the local CSV first, then fallback to URL if needed.

    Priority:
      1. Local `student_performance_data.csv`
      2. Remote URL
      3. Generate a sample dataset and save it locally
    """
    local = Path(__file__).resolve().parent / 'student_performance_data.csv'
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv"

    if local.exists():
        df = pd.read_csv(local)
        source = 'local file'
    else:
        try:
            df = pd.read_csv(url, sep=';')
            source = 'remote URL'
        except Exception as e:
            st.warning(f"Could not load dataset from URL: {e}\nFalling back to generated sample dataset.")
            np.random.seed(42)
            n_students = 395
            df = pd.DataFrame({
                'school': np.random.choice(['GP', 'MS'], n_students),
                'sex': np.random.choice(['F', 'M'], n_students),
                'age': np.random.randint(15, 23, n_students),
                'address': np.random.choice(['U', 'R'], n_students),
                'Pstatus': np.random.choice(['T', 'A'], n_students),
                'Medu': np.random.randint(0, 5, n_students),
                'Fedu': np.random.randint(0, 5, n_students),
                'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_students),
                'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_students),
                'reason': np.random.choice(['course', 'home', 'reputation', 'other'], n_students),
                'guardian': np.random.choice(['father', 'mother', 'other'], n_students),
                'traveltime': np.random.randint(1, 5, n_students),
                'studytime': np.random.randint(1, 5, n_students),
                'failures': np.random.randint(0, 4, n_students),
                'schoolsup': np.random.choice(['yes', 'no'], n_students),
                'famsup': np.random.choice(['yes', 'no'], n_students),
                'paid': np.random.choice(['yes', 'no'], n_students),
                'activities': np.random.choice(['yes', 'no'], n_students),
                'nursery': np.random.choice(['yes', 'no'], n_students),
                'higher': np.random.choice(['yes', 'no'], n_students),
                'internet': np.random.choice(['yes', 'no'], n_students),
                'romantic': np.random.choice(['yes', 'no'], n_students),
                'famrel': np.random.randint(1, 6, n_students),
                'freetime': np.random.randint(1, 6, n_students),
                'goout': np.random.randint(1, 6, n_students),
                'Dalc': np.random.randint(1, 6, n_students),
                'Walc': np.random.randint(1, 6, n_students),
                'health': np.random.randint(1, 6, n_students),
                'absences': np.random.randint(0, 93, n_students),
                'G1': np.random.randint(0, 21, n_students),
                'G2': np.random.randint(0, 21, n_students),
            })
            G1 = df['G1'].values
            G2 = df['G2'].values
            failures = df['failures'].values
            absences = df['absences'].values
            studytime = df['studytime'].values
            G3 = (G1 + G2) / 2 - (failures * 2) - (absences * 0.1) + (studytime * 0.5)
            G3 = np.clip(G3 + np.random.normal(0, 1.5, n_students), 0, 20).astype(int)
            df['G3'] = G3
            try:
                df.to_csv(local, index=False)
            except Exception:
                pass
            source = 'generated sample'

    if 'Pass' not in df.columns:
        df['Pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

    st.info(f"Dataset loaded from {source}. Shape: {df.shape}")
    return df

@st.cache_resource
def train_models(df):
    """Preprocess data and train all 3 models. Cached so not retrained on every click."""

    features = ['age', 'studytime', 'failures', 'absences',
                'G1', 'G2', 'Medu', 'Fedu', 'famrel', 'freetime', 'goout']

    X = df[features]
    y = df['Pass']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train all 3 models
    dt = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train_sc, y_train)

    # Get predictions
    results = {}
    for name, model, X_t in [
        ("Decision Tree", dt, X_test),
        ("Random Forest", rf, X_test),
        ("Naive Bayes",   nb, X_test_sc)
    ]:
        y_pred = model.predict(X_t)
        y_prob = model.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            'model'  : model,
            'y_pred' : y_pred,
            'y_prob' : y_prob,
            'acc'    : accuracy_score(y_test, y_pred),
            'prec'   : precision_score(y_test, y_pred, zero_division=0),
            'rec'    : recall_score(y_test, y_pred, zero_division=0),
            'f1'     : f1_score(y_test, y_pred, zero_division=0),
            'auc'    : auc(fpr, tpr),
            'fpr'    : fpr,
            'tpr'    : tpr,
            'cm'     : confusion_matrix(y_test, y_pred)
        }

    return results, X_test, y_test, features, X_train, dt


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')
    ax.tick_params(colors='#aaa')
    ax.xaxis.label.set_color('#aaa')
    ax.yaxis.label.set_color('#aaa')
    ax.title.set_color('#e0e0e0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3d5c')
    return fig, ax

def metric_card(label, value, color_class):
    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Data Mining Project")
    st.markdown("---")
    st.markdown("""
    **Domain:** Education  
    **Dataset:** UCI Student Performance  
    **Students:** 395 records  
    **Features:** 33 columns  
    """)
    st.markdown("---")
    st.markdown("### Algorithms Used")
    st.markdown("🌳 **Decision Tree** — splits data using yes/no questions")
    st.markdown("🌲 **Random Forest** — 100 trees voting together")
    st.markdown("🔵 **Naive Bayes** — probability-based predictions")
    st.markdown("---")
    st.markdown("### Metrics Evaluated")
    st.markdown("✅ Accuracy · Precision · Recall · F1 · AUC")
    st.markdown("---")
    st.caption("Dataset Link:")
    st.caption("archive.ics.uci.edu/ml/datasets/Student+Performance")


# ─────────────────────────────────────────────────────────────
# LOAD DATA & TRAIN
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading dataset and training models..."):
    df = load_data()
    results, X_test, y_test, features, X_train, dt_model = train_models(df)

MODEL_COLORS = {
    "Decision Tree": "#3498db",
    "Random Forest": "#2ecc71",
    "Naive Bayes":   "#e74c3c"
}


# ─────────────────────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:2.4rem; font-weight:800;
   background: linear-gradient(90deg,#3498db,#2ecc71);
   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
   margin-bottom:4px;'>
🎓 Student Performance Prediction
</h1>
<p style='text-align:center; color:#888; font-size:15px; margin-top:0;'>
Data Mining Project &nbsp;|&nbsp; Decision Tree · Random Forest · Naive Bayes
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_overview, tab_eda, tab_model, tab_compare = st.tabs([
    "📋 Dataset Overview",
    "📊 EDA",
    "📈 Model Results",
    "🏆 Comparison"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 : DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Total Students", "395", "blue"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Total Features", "33", "green"), unsafe_allow_html=True)
    with col3:
        pass_count = int(df['Pass'].sum())
        st.markdown(metric_card("Students Passed", str(pass_count), "green"), unsafe_allow_html=True)
    with col4:
        fail_count = int((df['Pass'] == 0).sum())
        st.markdown(metric_card("Students Failed", str(fail_count), "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.markdown("#### 📄 Raw Dataset (first 20 rows)")
        st.dataframe(df.head(20), use_container_width=True, height=350)
    with col_b:
        st.markdown("#### 📊 Column Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.notnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True, height=350)

    st.markdown("#### 📈 Basic Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown('<div class="info-box">ℹ️ <b>About this dataset:</b> Collected from two Portuguese secondary schools. '
                'Contains student grades, demographic info, social and school-related features. '
                'Target: G3 (final grade). We convert G3 → Pass (≥10) or Fail (<10).</div>',
                unsafe_allow_html=True)

    st.markdown("#### 🔗 Dataset Source")
    st.markdown("""
    - **Name:** UCI Student Performance Dataset
    - **URL:** https://archive.ics.uci.edu/ml/datasets/Student+Performance
    - **Direct CSV:** https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv
    - **Records:** 395 students | **Attributes:** 33 | **Target:** Pass/Fail (from G3)
    """)


# ══════════════════════════════════════════════════════════════
# TAB 2 : EDA
# ══════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>What is Exploratory Data Analysis (EDA)?</b><br>
    EDA is like being a detective with data! We explore the dataset to understand patterns, relationships, and insights
    that help us predict student performance. This section shows key visualizations that reveal what factors influence
    whether a student passes or fails their final exam. Even if you're new to this data, these charts will help you
    understand the story behind student success.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎯 Key Questions We're Answering")
    st.markdown("""
    - **What grades do students typically get?** (Grade Distribution)
    - **How many students pass vs fail?** (Pass/Fail Balance)
    - **Does studying more lead to better grades?** (Study Time Impact)
    - **Do absences hurt performance?** (Absence Effects)
    - **Which factors are most related to success?** (Correlations)
    - **How do past failures and family background matter?** (Additional Insights)
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 ────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Grade Distribution (G3)")
        fig, ax = dark_fig(6, 4)
        sns.histplot(df['G3'], bins=20, color='#3498db', kde=True, ax=ax)
        ax.axvline(x=10, color='#e74c3c', linestyle='--', linewidth=2, label='Pass line (10)')
        ax.set_xlabel("Final Grade (G3)")
        ax.set_ylabel("Number of Students")
        ax.set_title("Distribution of Final Grades")
        ax.legend(labelcolor='#aaa', facecolor='#1a1d2e', edgecolor='#3a3d5c')
        st.pyplot(fig)
        st.markdown('<div class="info-box">Most students score between 10–14. '
                    'Red dashed line = pass threshold (10).</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("##### Pass vs Fail Count")
        fig, ax = dark_fig(6, 4)
        counts = df['Pass'].value_counts().sort_index()
        bars = ax.bar(['Fail', 'Pass'], counts.values,
                      color=['#e74c3c', '#2ecc71'], alpha=0.85, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(int(bar.get_height())), ha='center', fontsize=14,
                    fontweight='bold', color='white')
        ax.set_title("Pass vs Fail Distribution")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)
        st.markdown(f'<div class="info-box">Pass: {pass_count} students ({pass_count/395*100:.1f}%) '
                    f'| Fail: {fail_count} students ({fail_count/395*100:.1f}%)</div>',
                    unsafe_allow_html=True)

    # ── Row 2 ────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("##### Study Time vs Final Grade")
        fig, ax = dark_fig(6, 4)
        bp = ax.boxplot(
            [df[df['studytime'] == i]['G3'].values for i in [1, 2, 3, 4]],
            patch_artist=True,
            medianprops=dict(color='white', linewidth=2)
        )
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ['whiskers', 'caps', 'fliers']:
            for item in bp[element]:
                item.set_color('#aaa')
        ax.set_xticklabels(['<2 hrs', '2-5 hrs', '5-10 hrs', '>10 hrs'])
        ax.set_xlabel("Weekly Study Time")
        ax.set_ylabel("Final Grade (G3)")
        ax.set_title("Study Time vs Final Grade")
        ax.axhline(y=10, color='white', linestyle=':', alpha=0.4)
        st.pyplot(fig)
        st.markdown('<div class="info-box">More study time → higher grades. '
                    'Students studying >5hrs/week score significantly better.</div>',
                    unsafe_allow_html=True)

    with c4:
        st.markdown("##### Absences vs Final Grade")
        fig, ax = dark_fig(6, 4)
        pass_mask = df['Pass'] == 1
        ax.scatter(df[~pass_mask]['absences'], df[~pass_mask]['G3'],
                   color='#e74c3c', alpha=0.5, label='Fail', s=30)
        ax.scatter(df[pass_mask]['absences'],  df[pass_mask]['G3'],
                   color='#2ecc71', alpha=0.5, label='Pass', s=30)
        ax.set_xlabel("Number of Absences")
        ax.set_ylabel("Final Grade (G3)")
        ax.set_title("Absences vs Final Grade")
        ax.axhline(y=10, color='white', linestyle=':', alpha=0.4)
        ax.legend(facecolor='#1a1d2e', edgecolor='#3a3d5c', labelcolor='#aaa')
        st.pyplot(fig)
        st.markdown('<div class="info-box">Students with many absences tend to fail. '
                    'High absence count clusters near low grades.</div>',
                    unsafe_allow_html=True)

    # ── Correlation Heatmap ───────────────────────────────────
    st.markdown("##### Correlation Heatmap")
    num_cols = ['age', 'studytime', 'failures', 'absences',
                'G1', 'G2', 'G3', 'Medu', 'Fedu', 'famrel', 'freetime', 'goout', 'Pass']
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=ax, mask=mask, annot_kws={"size": 9},
                linewidths=0.5, linecolor='#0f1117')
    ax.set_title("Feature Correlation Matrix", color='#e0e0e0', fontsize=13)
    ax.tick_params(colors='#aaa', labelsize=9)
    st.pyplot(fig)
    st.markdown('<div class="info-box">🔑 <b>Key insight:</b> G1 and G2 (previous grades) have very high positive '
                'correlation with G3 (final grade). Failures has strong negative correlation with Pass.</div>',
                unsafe_allow_html=True)

    # ── Failures vs Grade ─────────────────────────────────────
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("##### Past Failures vs Final Grade")
        fig, ax = dark_fig(6, 4)
        fail_data = [df[df['failures'] == i]['G3'].values for i in range(4)]
        bp = ax.boxplot(fail_data, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        fc = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        for patch, color in zip(bp['boxes'], fc):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ['whiskers', 'caps', 'fliers']:
            for item in bp[element]:
                item.set_color('#aaa')
        ax.set_xticklabels(['0 failures', '1 failure', '2 failures', '3 failures'])
        ax.set_ylabel("Final Grade (G3)")
        ax.set_title("Past Class Failures vs Final Grade")
        ax.axhline(y=10, color='white', linestyle=':', alpha=0.4)
        st.pyplot(fig)

    with c6:
        st.markdown("##### Parental Education vs Grade")
        fig, ax = dark_fig(6, 4)
        medu_means = df.groupby('Medu')['G3'].mean()
        fedu_means = df.groupby('Fedu')['G3'].mean()
        x = np.arange(5)
        ax.bar(x - 0.2, medu_means.values, 0.4, label="Mother's Edu", color='#9b59b6', alpha=0.8)
        ax.bar(x + 0.2, fedu_means.values, 0.4, label="Father's Edu", color='#3498db', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(['None', 'Primary', '5th-9th', 'Secondary', 'Higher'])
        ax.set_xlabel("Education Level")
        ax.set_ylabel("Average Final Grade")
        ax.set_title("Parental Education vs Average Grade")
        ax.legend(facecolor='#1a1d2e', edgecolor='#3a3d5c', labelcolor='#aaa')
        ax.axhline(y=10, color='white', linestyle=':', alpha=0.4)
        st.pyplot(fig)

    st.markdown("""
    <div class="info-box">
    <b>📊 EDA Summary: What We Learned About Student Success</b><br>
    <b>Top Predictors of Passing:</b> Previous grades (G1/G2) are the strongest indicators — students who did well before usually do well again.<br>
    <b>Study Habits Matter:</b> More weekly study time correlates with higher final grades.<br>
    <b>Attendance is Key:</b> High absences are a red flag for failing.<br>
    <b>Past Performance:</b> Students with previous class failures struggle more.<br>
    <b>Family Background:</b> Higher parental education levels are associated with better student performance.<br><br>
    <i>These insights guide our machine learning models to predict which students might need extra help to succeed!</i>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════
# TAB 4 : MODEL RESULTS
# ══════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-header">📈 Individual Model Results</div>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Select a model to inspect:",
        ["Decision Tree", "Random Forest", "Naive Bayes"],
        index=0,
        key="selected_model"
    )

    r = results[selected_model]
    color = MODEL_COLORS[selected_model]

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(metric_card("Accuracy",  f"{r['acc']*100:.1f}%",  "blue"),   unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Precision", f"{r['prec']*100:.1f}%", "green"),  unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Recall",    f"{r['rec']*100:.1f}%",  "orange"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("F1-Score",  f"{r['f1']*100:.1f}%",  "purple"), unsafe_allow_html=True)
    with col5:
        st.markdown(metric_card("AUC Score", f"{r['auc']:.4f}",       "red"),    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric explanations
    st.markdown("#### 📖 What each metric means")
    ex_c1, ex_c2 = st.columns(2)
    with ex_c1:
        tp = int(r['cm'][1][1])
        tn = int(r['cm'][0][0])
        fp = int(r['cm'][0][1])
        fn = int(r['cm'][1][0])
        total = tp + tn + fp + fn
        st.markdown(f"""
        <div class="info-box">
        <b>Accuracy = {r['acc']*100:.1f}%</b><br>
        {tp+tn} correct predictions out of {total} total.<br>
        Formula: (TP + TN) / Total = ({tp} + {tn}) / {total}
        </div>
        <div class="info-box">
        <b>Precision = {r['prec']*100:.1f}%</b><br>
        Of {tp+fp} students predicted Pass, {tp} actually passed.<br>
        Formula: TP / (TP + FP) = {tp} / ({tp} + {fp})
        </div>
        <div class="info-box">
        <b>Recall = {r['rec']*100:.1f}%</b><br>
        Of {tp+fn} students who actually passed, we caught {tp}.<br>
        Formula: TP / (TP + FN) = {tp} / ({tp} + {fn})
        </div>
        """, unsafe_allow_html=True)
    with ex_c2:
        st.markdown(f"""
        <div class="info-box">
        <b>F1-Score = {r['f1']*100:.1f}%</b><br>
        Balances Precision and Recall into one number.<br>
        Formula: 2 × (Precision × Recall) / (Precision + Recall)
        </div>
        <div class="info-box">
        <b>AUC Score = {r['auc']:.4f}</b><br>
        Area under the ROC curve. Range: 0.5 (useless) → 1.0 (perfect).<br>
        {r['auc']:.4f} means this model has {'excellent' if r['auc']>0.9 else 'good' if r['auc']>0.8 else 'fair'} discriminating ability.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════
# TAB 6 : COMPARISON
# ═══════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-header">🏆 Full Model Comparison</div>', unsafe_allow_html=True)

    # Summary table
    comparison_data = {
        'Model'     : list(MODEL_COLORS.keys()),
        'Accuracy'  : [f"{results[m]['acc']*100:.2f}%"  for m in MODEL_COLORS],
        'Precision' : [f"{results[m]['prec']*100:.2f}%" for m in MODEL_COLORS],
        'Recall'    : [f"{results[m]['rec']*100:.2f}%"  for m in MODEL_COLORS],
        'F1-Score'  : [f"{results[m]['f1']*100:.2f}%"   for m in MODEL_COLORS],
        'AUC Score' : [f"{results[m]['auc']:.4f}"       for m in MODEL_COLORS],
    }
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df.set_index('Model'), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart comparison
    st.markdown("#### 📊 Side-by-Side Metric Comparison")
    metrics_k = ['acc', 'prec', 'rec', 'f1', 'auc']
    metrics_l = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    x = np.arange(len(metrics_l))
    w = 0.25

    fig, ax = dark_fig(13, 6)
    for i, (name, color) in enumerate(MODEL_COLORS.items()):
        vals = [results[name][k] for k in metrics_k]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=name,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_l, fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score (0 to 1 scale)")
    ax.set_title("All Models — All Metrics Compared", fontsize=13)
    ax.legend(fontsize=11, facecolor='#1a1d2e', edgecolor='#3a3d5c', labelcolor='white')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    st.pyplot(fig)

    # All confusion matrices
    st.markdown("#### 🟦 All Confusion Matrices")
    c1, c2, c3 = st.columns(3)
    for col, (name, color) in zip([c1, c2, c3], MODEL_COLORS.items()):
        with col:
            fig, ax = dark_fig(4, 3.5)
            sns.heatmap(results[name]['cm'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Fail', 'Pass'],
                        yticklabels=['Fail', 'Pass'], ax=ax,
                        annot_kws={"size": 16, "weight": "bold"})
            ax.set_title(f"{name}\nAcc: {results[name]['acc']*100:.1f}%")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    # Winner announcement
    best_name = max(results, key=lambda m: results[m]['auc'])
    best = results[best_name]
    st.markdown(f"""
    <div class="winner-banner">
        <h2 style='color:#2ecc71; margin:0;'>🏆 Best Model: {best_name}</h2>
        <p style='color:#aaa; margin:8px 0 0 0;'>
        Accuracy: {best['acc']*100:.2f}% &nbsp;|&nbsp;
        Precision: {best['prec']*100:.2f}% &nbsp;|&nbsp;
        Recall: {best['rec']*100:.2f}% &nbsp;|&nbsp;
        F1: {best['f1']*100:.2f}% &nbsp;|&nbsp;
        AUC: {best['auc']:.4f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric explanations summary
    st.markdown("#### 📖 Quick Reference: What Each Metric Means")
    st.markdown("""
    | Metric | Full Name | What it measures | Best value |
    |---|---|---|---|
    | **Accuracy** | Accuracy | % of all predictions that were correct | 1.00 (100%) |
    | **Precision** | Positive Predictive Value | Of predicted Pass, how many actually passed | 1.00 |
    | **Recall** | Sensitivity / TPR | Of actual Pass, how many did we find | 1.00 |
    | **F1-Score** | F1 | Balance between Precision and Recall | 1.00 |
    | **AUC** | Area Under ROC Curve | Overall separation ability of model | 1.00 |
    """)

    st.markdown("""
    <div class="info-box">
    <b>Why Random Forest usually wins:</b><br>
    It trains 100 different decision trees on random subsets of data.
    Each tree votes, and the majority vote wins. This "wisdom of crowds" approach
    reduces errors and overfitting compared to a single Decision Tree.
    Naive Bayes is simplest and fastest but its "independent features" assumption
    makes it less accurate when features are correlated (which they are here — G1 and G2 are related).
    </div>
    """, unsafe_allow_html=True)
