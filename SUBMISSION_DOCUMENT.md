# BITS Pilani — Work Integrated Learning Programmes (WILP)
## M.Tech (AIML) | Machine Learning (Assignment - 2)

---

### Student Submission Details
- **Student Name**: Dhanush P
- **BITS Student ID**: 2026aj05570
- **Course**: Machine Learning
- **Program**: M.Tech (AIML)
- **Submission Date**: 08-November-2026

---

## 1. Mandatory Submission Links

### 1.1 GitHub Repository Link
**Repository URL**: [https://github.com/zoroyadav/Connect4-ML-Assignment](https://github.com/zoroyadav/Connect4-ML-Assignment)

**Repository Contents**:
- `app.py`: Streamlit interactive web application
- `requirements.txt`: Package dependency manifest
- `README.md`: Structured assignment documentation and analysis
- `test_data.csv`: Test partition used in experiments (3,000 instances with 42 features and ground-truth target)
- `train_models.py` & `train_models.ipynb`: Complete source code for data loading, preprocessing, model training, evaluation, and pipeline serialization
- `model/`: Directory storing all five trained classifier pipelines (`.joblib`), metadata, and source scripts:
  - `logistic_regression.joblib`
  - `decision_tree.joblib`
  - `knn.joblib`
  - `naive_bayes.joblib`
  - `svm.joblib`

### 1.2 Live Streamlit App Link
**Streamlit App URL**: [https://connect4-ml-classifier.streamlit.app](https://connect4-ml-assignment-baq7d5sryrcass8wxpy6vs.streamlit.app/)

---

## 2. BITS Virtual Lab Execution Proof

*(Paste your ONE full-screen screenshot showing execution of the assignment inside the BITS Virtual Lab environment here)*

> **[INSERT SCREENSHOT OF BITS VIRTUAL LAB EXECUTION HERE]**

---

## 3. GitHub README Content

### a. Problem Statement
Connect-4 is a two-player zero-sum board game played on a 6-row by 7-column vertical grid. Players alternately drop coloured tokens from the top into one of seven columns, aiming to form an unbroken horizontal, vertical, or diagonal line of four pieces.

The objective of this assignment is to:
1. Develop, train, evaluate, and compare **five supervised machine learning classification models** on the UCI Connect-4 opening database.
2. Predict the game outcome for Player 1 ('x') into three distinct classes: **`win`**, **`loss`**, or **`draw`**, given an 8-ply legal game position where neither player has won yet and the next move is not forced.
3. Compute and contrast six standard evaluation metrics: **Accuracy, AUC Score (One-vs-Rest weighted), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**.
4. Deploy an interactive **Streamlit web application** capable of loading the saved model pipelines, ingesting uploaded test data, and computing evaluation metrics without retraining.

---

### b. Dataset Description
- **Dataset**: Connect-4 Opening Database
- **Repository**: UCI Machine Learning Repository (Dataset ID: 26)
- **Total Instances**: 67,557 legal 8-ply positions
- **Feature Count**: 42 categorical features corresponding to the 42 squares on the 6×7 grid:
  - Columns: `a`, `b`, `c`, `d`, `e`, `f`, `g` (left to right)
  - Rows: `1`, `2`, `3`, `4`, `5`, `6` (bottom to top)
  - Attributes: `a1` through `g6`
- **Feature Attribute Values**:
  - `x`: Square occupied by Player 1 (first player)
  - `o`: Square occupied by Player 2 (second player)
  - `b`: Blank / unoccupied square
- **Target Variable (`class`)**: 3 discrete outcome classes:
  - `win`: Player 1 will win with optimal play (44,473 instances / 65.83%)
  - `loss`: Player 1 will lose with optimal play (16,635 instances / 24.62%)
  - `draw`: Game ends in a draw with optimal play (6,449 instances / 9.55%)
- **Missing Values**: 0 (no missing values)
- **Experimental Setup**: A stratified 80-20 partition (12,000 training instances, 3,000 test instances) was used for model training and benchmark verification. Categorical board attributes were encoded via `OneHotEncoder(handle_unknown='ignore')` embedded inside scikit-learn `Pipeline` objects.

---

### c. GitHub Repository Link
[https://github.com/zoroyadav/Connect4-ML-Assignment](https://github.com/zoroyadav/Connect4-ML-Assignment)

---

### d. Models Used & Comparison Table

#### Evaluation Metrics Summary Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7547 | 0.8488 | 0.7149 | 0.7547 | 0.7152 | 0.4580 |
| **Decision Tree** | 0.7150 | 0.7802 | 0.6914 | 0.7150 | 0.7002 | 0.3895 |
| **KNN** | 0.7550 | 0.8377 | 0.7238 | 0.7550 | 0.7251 | 0.4559 |
| **Naive Bayes** | 0.7213 | 0.8001 | 0.6847 | 0.7213 | 0.6809 | 0.3612 |
| **Support Vector Machine** | **0.7943** | **0.9066** | **0.7764** | **0.7943** | **0.7593** | **0.5521** |

*Evaluation Notes: All metrics were calculated on the identical 3,000-instance unseen test set (`test_data.csv`). AUC is computed using One-vs-Rest (OvR) multiclass formulation with weighted averaging. Precision, Recall, and F1 scores are weighted across all classes.*

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Logistic Regression demonstrated a solid linear baseline with an Accuracy of 75.47% and an AUC of 0.8488. The one-hot encoded representation allowed the model to estimate weights for each board cell; it successfully captured the strategic importance of occupying central columns (`c`, `d`, `e`) which offer the greatest number of potential 4-in-a-row connections. However, because its decision boundary is inherently planar in feature space, it is fundamentally incapable of modeling multi-cell diagonal dependencies or complex trapping geometries without manual feature crossing. |
| **Decision Tree** | Decision Tree reached 71.50% Accuracy, 0.7802 AUC, and an MCC of 0.3895. Decision trees partition space using greedy, axis-aligned orthogonal splits on individual squares. In Connect-4, winning lines occur horizontally, vertically, and diagonally; splitting on one board square in isolation provides minimal mutual information regarding whether an entire 4-piece diagonal alignment will form. Single trees suffer from high variance and local minima on such combinatorial problems. |
| **KNN** | K-Nearest Neighbors achieved strong predictive capability with 75.50% Accuracy, 0.8377 AUC, and an MCC of 0.4559. By evaluating distance-weighted nearest neighbors ($k=7$) in the one-hot encoded space, KNN effectively clustered game positions that share identical structural piece arrangements. However, KNN does not learn a parametric representation; its inference complexity scales with dataset size, requiring distance calculations against all training vectors during evaluation. |
| **Naive Bayes** | Multinomial Naive Bayes yielded 72.13% Accuracy and an AUC of 0.8001 with an MCC of 0.3612. Naive Bayes assumes that board squares are conditionally independent given the game outcome. In Connect-4, this assumption is strongly violated because gravity dictates piece placement (pieces cannot hover) and victory depends strictly on contiguous sequences of four adjacent pieces. While independence violations caused slightly depressed precision, the algorithm trained virtually instantaneously and provided a decent baseline. |
| **Support Vector Machine** | Support Vector Machine (SVM) with an RBF kernel was the top-performing model across **every single metric**: Accuracy of 79.43%, AUC of 0.9066, Precision of 77.64%, Recall of 79.43%, F1 Score of 75.93%, and MCC of 0.5521. By projecting the categorical board configurations into an infinite-dimensional reproducing kernel Hilbert space via the RBF kernel, SVM effectively mapped non-linear spatial relationships and diagonal winning configurations into a linearly separable space with maximum geometric margin separation. |

---

### Overall, Winner for the dataset?

**Support Vector Machine (SVM)** is the clear **winner** for the Connect-4 dataset.

1. **Superior Discriminative Power**: It established the benchmark across all metrics (Accuracy: **79.43%**, AUC: **0.9066**, MCC: **0.5521**), outperforming linear, tree-based, and probabilistic baselines.
2. **Handling Non-Linear Spatial Dependencies**: The RBF kernel captures spatial adjacency and diagonal 4-in-a-row alignments without requiring manual polynomial feature engineering.
3. **Generalization Robustness**: The maximum-margin hyperplanes provide superior resistance to overfitting on complex tactical game states.
