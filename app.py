import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# Page configuration
st.set_page_config(
    page_title="Connect-4 ML Classifier | BITS Assignment 2",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .board-cell {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        text-align: center;
        border-radius: 50%;
        margin: 2px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "Support Vector Machine (SVM)": "model/svm.joblib",
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
}

MODEL_DISPLAY_NAMES = {
    "Support Vector Machine (SVM)": "Support Vector Machine",
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "K-Nearest Neighbors (KNN)": "KNN",
    "Naive Bayes": "Naive Bayes"
}

@st.cache_resource
def load_all_models():
    """Load all 5 pre-trained pipelines with caching."""
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            alt_path = os.path.join(os.path.dirname(__file__), path)
            if os.path.exists(alt_path):
                models[name] = joblib.load(alt_path)
    return models

@st.cache_data
def load_default_data():
    """Load default test_data.csv if available."""
    default_path = "test_data.csv"
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None

def compute_metrics(model, X_test, y_test):
    """Compute all 6 required evaluation metrics on the test data."""
    y_pred = model.predict(X_test)
    
    # Calculate probability estimates
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    if y_proba is not None:
        try:
            auc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            auc_val = 0.0
    else:
        auc_val = 0.0

    return {
        "Accuracy": acc,
        "AUC": auc_val,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

def main():
    st.markdown('<div class="main-header">Connect-4 Classification & Evaluation Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">BITS Pilani — M.Tech (AIML) Machine Learning Assignment 2</div>', unsafe_allow_html=True)

    # Load pre-trained models
    models = load_all_models()
    if not models:
        st.error("No models found in `model/`. Please ensure trained .joblib files are present.")
        return

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Controls & Upload")
        st.markdown("**Step a: Upload Test Dataset**")
        uploaded_file = st.file_uploader(
            "Upload test CSV file (Features + Target column)",
            type=["csv"],
            help="Upload test dataset containing 42 board features (a1..g6) and target column ('class' or 'target')."
        )
        
        use_default = False
        if uploaded_file is None:
            st.info("No CSV uploaded yet. You can use the pre-generated test dataset.")
            if st.button("📁 Load Default Test Data (test_data.csv)", use_container_width=True):
                use_default = True

        st.markdown("---")
        st.markdown("**Step b: Model Selection**")
        model_options = list(MODEL_FILES.keys()) + ["📊 Compare All 5 Models"]
        selected_model_name = st.selectbox(
            "Select Classification Model:",
            options=model_options,
            index=0
        )

        st.markdown("---")
        st.markdown("### ℹ️ Dataset Information")
        st.markdown("""
        - **Dataset**: UCI Connect-4 Database
        - **Board Size**: 6 rows × 7 columns (42 cells)
        - **Feature Values**: `x` (player 1), `o` (player 2), `b` (blank)
        - **Target Classes**: `win`, `loss`, `draw`
        - **Total Features**: 42
        """)

    # Load DataFrame
    df = None
    data_source_label = ""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            data_source_label = f"Uploaded File: `{uploaded_file.name}`"
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return
    else:
        default_df = load_default_data()
        if default_df is not None:
            df = default_df
            data_source_label = "Default Test Dataset (`test_data.csv`)"
        else:
            st.warning("Please upload a test dataset CSV to proceed.")
            return

    # Detect Target Column
    target_col = None
    for candidate in ["class", "target", "outcome", "label"]:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        # Default to last column
        target_col = df.columns[-1]

    X_test = df.drop(columns=[target_col]).astype(str)
    y_test = df[target_col].astype(str)

    # Data Overview Expandable
    with st.expander(f"📋 Test Dataset Preview ({data_source_label}) - {len(df)} Instances", expanded=False):
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Total Test Rows", len(df))
        col_info2.metric("Feature Count", X_test.shape[1])
        col_info3.metric("Target Column", target_col)
        
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("**Class Distribution in Test Set:**")
        dist_df = y_test.value_counts().reset_index()
        dist_df.columns = ["Class", "Count"]
        dist_df["Percentage"] = (dist_df["Count"] / len(y_test) * 100).round(2).astype(str) + "%"
        st.dataframe(dist_df, use_container_width=True)

    # Check feature count
    if X_test.shape[1] != 42:
        st.warning(f"Warning: Expected 42 board features, but found {X_test.shape[1]}. Models were trained on 42 features (a1..g6).")

    # MODE 1: COMPARE ALL 5 MODELS
    if selected_model_name == "📊 Compare All 5 Models":
        st.markdown("## 📊 Comparison of All 5 Classification Models")
        st.markdown("Evaluating all five saved models on the uploaded test dataset:")

        summary_rows = []
        progress_bar = st.progress(0)
        
        for idx, (m_key, model_pipe) in enumerate(models.items()):
            res = compute_metrics(model_pipe, X_test, y_test)
            display_name = MODEL_DISPLAY_NAMES.get(m_key, m_key)
            summary_rows.append({
                "ML Model Name": display_name,
                "Accuracy": round(res["Accuracy"], 4),
                "AUC": round(res["AUC"], 4),
                "Precision": round(res["Precision"], 4),
                "Recall": round(res["Recall"], 4),
                "F1": round(res["F1"], 4),
                "MCC": round(res["MCC"], 4)
            })
            progress_bar.progress((idx + 1) / len(models))
        
        progress_bar.empty()
        comp_df = pd.DataFrame(summary_rows)
        
        # Display Official Table (exact format requested in Step 5 - Step d)
        st.markdown("### Mandatory Metric Comparison Table")
        st.dataframe(
            comp_df.style.highlight_max(
                subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                color="#C6F6D5"
            ),
            use_container_width=True
        )

        # Plotly / Matplotlib Comparison Bar Chart
        st.markdown("### Metric Comparison Visualizer")
        melted_df = comp_df.melt(
            id_vars=["ML Model Name"],
            value_vars=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            var_name="Metric",
            value_name="Score"
        )
        
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(
            data=melted_df,
            x="ML Model Name",
            y="Score",
            hue="Metric",
            palette="crest",
            ax=ax
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
        ax.set_xlabel("Machine Learning Model", fontsize=11)
        ax.set_title("Performance Metrics across all 5 Classifiers", fontsize=13, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Winner Announcement Card
        best_model = comp_df.loc[comp_df["Accuracy"].idxmax()]["ML Model Name"]
        st.success(f"🏆 **Overall Winner for Connect-4 Dataset:** **{best_model}** achieved the highest score across Accuracy, AUC, and MCC!")

    # MODE 2: SINGLE MODEL DEEP-DIVE
    else:
        active_pipe = models[selected_model_name]
        st.markdown(f"## Model Analysis: **{selected_model_name}**")
        
        # Compute metrics
        with st.spinner("Calculating predictions and evaluation metrics..."):
            metrics = compute_metrics(active_pipe, X_test, y_test)
        
        st.markdown("### Step c: Display of Evaluation Metrics")
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{metrics['Accuracy']:.4f}</div>
                <div style="font-size:0.8rem; color:#10B981;">{metrics['Accuracy']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AUC Score</div>
                <div class="metric-value">{metrics['AUC']:.4f}</div>
                <div style="font-size:0.8rem; color:#6366F1;">OvR Weighted</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{metrics['Precision']:.4f}</div>
                <div style="font-size:0.8rem; color:#64748B;">Weighted</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{metrics['Recall']:.4f}</div>
                <div style="font-size:0.8rem; color:#64748B;">Weighted</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{metrics['F1']:.4f}</div>
                <div style="font-size:0.8rem; color:#EC4899;">Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MCC Score</div>
                <div class="metric-value">{metrics['MCC']:.4f}</div>
                <div style="font-size:0.8rem; color:#8B5CF6;">Correlation</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Step d: Confusion Matrix & Classification Report")
        
        diag_col1, diag_col2 = st.columns([1, 1])
        classes = list(active_pipe.classes_)
        
        with diag_col1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, metrics["y_pred"], labels=classes)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax_cm,
                cbar=False
            )
            ax_cm.set_xlabel("Predicted Outcome", fontsize=11, fontweight="bold")
            ax_cm.set_ylabel("Actual Outcome", fontsize=11, fontweight="bold")
            ax_cm.set_title(f"Confusion Matrix ({selected_model_name})", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with diag_col2:
            st.markdown("#### Classification Report")
            cr_dict = classification_report(
                y_test, metrics["y_pred"], target_names=classes, output_dict=True, zero_division=0
            )
            cr_df = pd.DataFrame(cr_dict).transpose().round(4)
            st.dataframe(cr_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]), use_container_width=True)

            if metrics["y_proba"] is not None:
                st.markdown("#### Multiclass ROC-AUC Curves")
                y_bin = label_binarize(y_test, classes=classes)
                fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                for i, class_label in enumerate(classes):
                    if y_bin.shape[1] > 1:
                        fpr, tpr, _ = roc_curve(y_bin[:, i], metrics["y_proba"][:, i])
                        roc_auc = auc(fpr, tpr)
                        ax_roc.plot(fpr, tpr, label=f"{class_label} (AUC = {roc_auc:.3f})")
                ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6)
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("One-vs-Rest ROC Curve")
                ax_roc.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_roc)
                plt.close(fig_roc)

        # Custom Bonus Feature: Interactive Connect-4 Board Visualizer
        st.markdown("---")
        st.markdown("### 🎲 Interactive Connect-4 Board Inspector")
        st.markdown("Select an individual board position from the test set to visualize the physical 6x7 grid, inspect Player 1 (`x`) vs Player 2 (`o`) placements, and view model predictions:")

        sample_idx = st.slider("Select Sample Index from Test Set:", min_value=0, max_value=len(X_test)-1, value=0)
        selected_sample = X_test.iloc[sample_idx]
        actual_val = y_test.iloc[sample_idx]
        pred_val = metrics["y_pred"][sample_idx]
        
        board_col1, board_col2 = st.columns([1, 1])
        with board_col1:
            st.markdown(f"**Board Representation for Sample #{sample_idx}:**")
            # Connect 4 grid is 6 rows (6 down to 1), 7 columns (a to g)
            grid_html = "<div style='background-color:#1E40AF; padding:15px; border-radius:12px; display:inline-block;'>"
            for r in [6, 5, 4, 3, 2, 1]:
                grid_html += "<div style='display:flex; justify-content:center;'>"
                for c in ["a", "b", "c", "d", "e", "f", "g"]:
                    cell_name = f"{c}{r}"
                    cell_val = selected_sample.get(cell_name, "b")
                    if cell_val == "x":
                        # Red checker for Player 1
                        grid_html += "<span class='board-cell' style='background-color:#EF4444; color:white;'>X</span>"
                    elif cell_val == "o":
                        # Yellow checker for Player 2
                        grid_html += "<span class='board-cell' style='background-color:#FBBF24; color:black;'>O</span>"
                    else:
                        # Empty slot
                        grid_html += "<span class='board-cell' style='background-color:#DBEAFE; color:#94A3B8;'>·</span>"
                grid_html += "</div>"
            grid_html += "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)
            st.caption("🔴 Red = Player 1 ('x') | 🟡 Yellow = Player 2 ('o') | ⚪ White = Empty ('b')")

        with board_col2:
            st.markdown("#### Sample Predictions")
            st.markdown(f"- **Actual Outcome:** `{actual_val}`")
            if pred_val == actual_val:
                st.success(f"- **Predicted Outcome:** `{pred_val}` (Correct ✅)")
            else:
                st.error(f"- **Predicted Outcome:** `{pred_val}` (Incorrect ❌)")

            if metrics["y_proba"] is not None:
                st.markdown("**Predicted Class Probabilities:**")
                proba_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": [f"{p*100:.2f}%" for p in metrics["y_proba"][sample_idx]]
                })
                st.dataframe(proba_df, use_container_width=True)

if __name__ == "__main__":
    main()
