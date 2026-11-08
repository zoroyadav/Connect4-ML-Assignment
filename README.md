# Connect-4 Game Outcome Prediction & Evaluation Studio

**BITS Pilani — Work Integrated Learning Programmes (WILP)**  
**M.Tech (AIML) | Machine Learning — Assignment 2**

---

## a. Problem Statement

Connect-4 is a two-player zero-sum game played on a vertical 6-row by 7-column grid where players take turns dropping colored discs from the top into a seven-column, six-row vertically suspended grid. The objective is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs. 

The objective of this assignment is to:
1. Develop, train, evaluate, and compare **five supervised machine learning classification models** on the UCI Connect-4 opening database.
2. Formulate the task as a multi-class classification problem predicting the final game outcome for Player 1 ('x') into three distinct classes: **`win`**, **`loss`**, or **`draw`**, given an 8-ply legal game position where neither player has won yet and the next move is not forced.
3. Compute and contrast six standard evaluation metrics: **Accuracy, AUC Score (One-vs-Rest weighted), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**.
4. Build and deploy an interactive web application using **Streamlit Community Cloud** that loads the serialized model pipelines and evaluates uploaded test data without retraining.

---

## b. Dataset Description

- **Dataset Name**: Connect-4 Opening Database
- **Source**: UCI Machine Learning Repository (Dataset ID: 26)
- **Total Instances**: 67,557 legal 8-ply positions
- **Feature Count**: 42 categorical features corresponding to the 42 squares on the 6×7 grid:
  - Columns: `a`, `b`, `c`, `d`, `e`, `f`, `g` (from left to right)
  - Rows: `1`, `2`, `3`, `4`, `5`, `6` (from bottom to top)
  - Features: `a1`, `a2`, ..., `a6`, `b1`, ..., `g6`
- **Feature Attribute Values**:
  - `x`: Square occupied by Player 1 (first player)
  - `o`: Square occupied by Player 2 (second player)
  - `b`: Blank / unoccupied square
- **Target Variable (`class`)**: Multi-class classification target with 3 classes:
  - `win`: Player 1 will win with optimal play (44,473 instances / 65.83%)
  - `loss`: Player 1 will lose with optimal play (16,635 instances / 24.62%)
  - `draw`: Game terminates in a draw with optimal play (6,449 instances / 9.55%)
- **Missing Values**: None (0 missing entries)
- **Experimental Setup**: A stratified representative split of 15,000 instances (12,000 training samples and 3,000 testing samples) was utilized to maintain the exact class distribution while ensuring fast inference, low model sizes (<1 MB), and seamless execution within Streamlit Cloud free-tier memory boundaries.

---

## c. Github Repository Link

- **GitHub Repository**: [https://github.com/dhanush-bits/connect4-ml-assignment](https://github.com/dhanush-bits/connect4-ml-assignment)
*(Replace with your personal repository URL upon pushing)*

### Repository Directory Structure
```text
project-folder/
│-- app.py                   # Streamlit interactive web application
│-- requirements.txt         # Project dependencies for cloud deployment
│-- README.md                # Comprehensive project documentation & report
│-- test_data.csv            # Test dataset (3,000 instances with 42 features + class)
│-- train_models.py          # Standalone reproducible training script
│-- train_models.ipynb       # Jupyter Notebook with EDA, training & evaluation
│-- model/                   # Serialized model artifacts & training source
    │-- logistic_regression.joblib  # Trained Logistic Regression pipeline
    │-- decision_tree.joblib        # Trained Decision Tree pipeline
    │-- knn.joblib                  # Trained K-Nearest Neighbors pipeline
    │-- naive_bayes.joblib          # Trained Multinomial Naive Bayes pipeline
    │-- svm.joblib                  # Trained Support Vector Machine pipeline
    │-- train_models.py             # Source training code
    │-- train_models.ipynb          # Jupyter Notebook source
    │-- comparison_table.csv        # Computed metrics comparison table
    │-- metrics.json                # Precomputed validation metrics metadata
```

---

## d. Models Used & Evaluation

### Comparison Table

The following comparison table presents the six evaluation metrics calculated for all five machine learning models on the exact same test dataset (`test_data.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7547 | 0.8488 | 0.7149 | 0.7547 | 0.7152 | 0.4580 |
| **Decision Tree** | 0.7150 | 0.7802 | 0.6914 | 0.7150 | 0.7002 | 0.3895 |
| **KNN** | 0.7550 | 0.8377 | 0.7238 | 0.7550 | 0.7251 | 0.4559 |
| **Naive Bayes** | 0.7213 | 0.8001 | 0.6847 | 0.7213 | 0.6809 | 0.3612 |
| **Support Vector Machine** | **0.7943** | **0.9066** | **0.7764** | **0.7943** | **0.7593** | **0.5521** |

*Note: AUC is computed using One-vs-Rest (OvR) with weighted averaging across all 3 classes. Precision, Recall, and F1 scores are weighted by class support to handle class imbalance.*

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Logistic Regression demonstrated a strong linear baseline with an Accuracy of 75.47% and an AUC of 0.8488. Using one-hot encoded board representations, the L2-regularized logistic model successfully learns the first-order strategic importance of controlling central columns (columns c, d, and e) and early vertical momentum. However, because it operates with a linear decision boundary, it cannot model non-linear spatial interactions or diagonal multi-piece alignment traps without extensive manual polynomial feature engineering. |
| **Decision Tree** | Decision Tree achieved an Accuracy of 71.50% and an AUC of 0.7802. While offering transparent, interpretable hierarchical rules, single decision trees suffer from greedy, axis-aligned recursive binary splitting. In Connect-4, winning lines are formed horizontally, vertically, and diagonally across 42 interrelated board squares; isolated orthogonal splits struggle to capture these multi-directional diagonal symmetries without suffering from variance or local optima. |
| **KNN** | K-Nearest Neighbors delivered strong performance with 75.50% Accuracy, 0.8377 AUC, and an MCC of 0.4559. By evaluating distance-weighted neighbors in the one-hot encoded state space ($k=7$), KNN effectively identifies similar board patterns from past games. Its primary drawbacks are non-parametric memory storage and test-time latency, as calculating distances against all training exemplars scales linearly with sample size. |
| **Naive Bayes** | Multinomial Naive Bayes yielded an Accuracy of 72.13% and an AUC of 0.8001. Naive Bayes relies on the strict conditional independence assumption between features given the class label. In Connect-4, board positions are strongly interdependent due to gravity constraints (discs cannot hover) and contiguous 4-in-a-row spatial patterns. Despite this theoretical limitation, the model trained in sub-second time and offered a surprisingly competitive baseline. |
| **Support Vector Machine** | Support Vector Machine (SVM) with an RBF kernel achieved the highest performance across **all six evaluation metrics** with an **Accuracy of 79.43%**, **AUC of 0.9066**, **Precision of 0.7764**, **Recall of 0.7943**, **F1 Score of 0.7593**, and **MCC of 0.5521**. The non-linear Radial Basis Function (RBF) kernel implicitly projects the board features into an infinite-dimensional reproducing kernel Hilbert space where complex spatial patterns, diagonal win alignments, and multi-piece combinations become linearly separable with a maximal margin. |

---

### Overall, Winner for your dataset?

**Support Vector Machine (SVM)** is the undisputed **winner** for the Connect-4 dataset. 
- It achieved the top score across **every single metric**: highest Accuracy (79.43%), highest AUC (0.9066), highest Precision (77.64%), highest Recall (79.43%), highest F1 Score (75.93%), and highest Matthews Correlation Coefficient (0.5521).
- The non-linear RBF kernel is uniquely suited to model the spatial, 2D geometric win conditions (horizontal, vertical, and diagonal lines) inherent to Connect-4 board mechanics without succumbing to the independence limitations of Naive Bayes or the high variance of individual decision trees.

---

## Step 6: How to Run Locally and Deploy

### 1. Local Execution
```bash
# Clone the repository
git clone https://github.com/dhanush-bits/connect4-ml-assignment.git
cd connect4-ml-assignment

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

### 2. Deployment on Streamlit Community Cloud
1. Push your repository to your GitHub account.
2. Navigate to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click **"New App"**.
4. Select your repository (`connect4-ml-assignment`), branch (`main`), and specify the main file path as `app.py`.
5. Click **Deploy**. The app will be live and interactive within 1–2 minutes!

---

## Streamlit Application Features Checklist
- [x] **Dataset upload option (CSV)**: Allows user to upload any test CSV containing 42 board features + target, with fallback to pre-loaded `test_data.csv`.
- [x] **Model selection dropdown**: Enables switching between all 5 individual models plus an interactive "Compare All 5 Models" mode.
- [x] **Display of evaluation metrics**: Shows Accuracy, AUC, Precision, Recall, F1 Score, and MCC in high-contrast KPI cards.
- [x] **Confusion matrix & classification report**: Visualized using styled heatmaps, per-class metrics, and Multiclass ROC Curves.
- [x] **Custom Connect-4 Board Visualizer**: Allows selecting any board position and visualizes the physical 6×7 grid with red (Player 1) and yellow (Player 2) discs alongside live predictions.
