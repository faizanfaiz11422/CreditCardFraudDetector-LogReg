# 🧠 Credit Card Fraud Detection Using Logistic Regression

A robust, modular machine learning pipeline for binary classification of credit card transactions. This project blends data preprocessing, scaling, class-wise sampling, regularized logistic regression training, and CSV logging—topped with a PyQt5 UI.

## 📦 Structure
Logistic Regression/
├── Logistic_regression.py     # Core logic: Data prep, scaling, training
├── prediction.py              # Prediction pipeline (loaded weights & bias)
├── creditcard.csv             # Non-fraudulent transactions
├── creditcard1.csv            # Fraudulent transactions
├── README.md                  # This doc

## 📊 Dataset Overview

- **Source:** Anonymized credit card transaction data
- **Classes:**
  - `Class 0`: Non-fraudulent → `creditcard.csv`
  - `Class 1`: Fraudulent → `creditcard1.csv`
- **Challenge:** Heavy class imbalance handled via manual undersampling and separation

## ⚙️ How It Works

### 💾 Data Preprocessing

- Imports class-wise CSVs
- Converts data → NumPy → Pandas → shuffled samples
- Separates features and labels
- Scales features using Min-Max normalization
- Splits into Train / Cross-Validation / Test sets (60/20/20)

### 📈 Training Logic

- Implements:
  - Sigmoid activation
  - Cost function (w/ optional L2 regularization)
  - Gradient descent with regularized weight updates
- Saves weights and bias (`theta.npy`, `bias.npy`) for inference

### 🧪 Testing

- Preprocessed test sets saved as:
  - `test_x.npy`
  - `test_y.npy`
- Run inference via `prediction.py`

---

## 🧮 Key Code Snippets

```python
def sigmoid(X, theta, B):
    return 1 / (1 + np.exp(-(np.dot(theta, X.T) + B)))

def cost(X, y, theta):
    # Regularized and unregularized cost computation
    ...

def gradient_descent(X, y, theta, B, alpha, iterations):
    # Weight updates via backpropagation
    ...
```
## 🖼️ UI & Visualization

- Developed using **PyQt5** to create an intuitive GUI for fraud prediction.
- Users can input transaction details directly through the interface.
- Predictions are evaluated using pre-trained logistic regression parameters.
- **CSV logging** stores each prediction with timestamps for audit and analysis.
- **Planned enhancements**:
  - 📉 ROC curve visualization to assess model sensitivity and specificity.
  - 📊 Class distribution charts for exploring data imbalance and patterns.

---

## ⚙️ Installation & Usage

### 🔧 Installation

```bash
pip install numpy pandas scikit-learn PyQt5
```
### 🏁 Train the Model
```bash
python Logistic_regression.py
```
### 🧠 Run Predictions
``` bash
python prediction.py
```
### 🚀 Potential Enhancements

- Switch from manual cost calculation to scikit-learn or statsmodels for benchmarking.

- Implement ROC-AUC and Precision-Recall metrics.

- Add an LLM-based UI overlay for fraud pattern explanation.

- Create a FastAPI endpoint for real-time deployment.

👨‍💻 Author
[**M Faizan Faiz**](https://github.com/faizanfaiz11422)

