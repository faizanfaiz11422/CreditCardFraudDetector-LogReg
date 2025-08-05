🧠 Credit Card Fraud Detection Using Logistic Regression
A robust, modular machine learning pipeline for binary classification of credit card transactions. This project blends data preprocessing, scaling, class-wise sampling, regularized logistic regression training, and CSV logging—topped with a PyQt5 UI.

📦 Structure
Logistic Regression/
├── Logistic_regression.py     # Core logic: Data prep, scaling, training
├── prediction.py              # Prediction pipeline (loaded weights & bias)
├── creditcard.csv             # Non-fraudulent transactions
├── creditcard1.csv            # Fraudulent transactions
├── README.md                  # This doc

📊 Dataset Overview
Source: Anonymized credit card transaction data

Classes:

Class 0: Non-fraudulent (from creditcard.csv)

Class 1: Fraudulent (from creditcard1.csv)

Challenges: High class imbalance, requiring manual undersampling and separation.

⚙️ How It Works
💾 Data Preprocessing
Class-wise CSV import (creditcard.csv, creditcard1.csv)

Conversion to NumPy ➝ Pandas ➝ shuffled samples

Separation of features and labels

Manual scaling using Min-Max normalization for each feature

Split into Train / Cross-Validation / Test sets (60/20/20 split)

📈 Training Logic
Manual implementation of:

Sigmoid function

Cost function with optional L2 regularization

Gradient descent with regularized weight updates

Saving trained weights (theta.npy) and bias (bias.npy) for deployment

🧪 Testing
Preprocessed test sets are saved as NumPy arrays (test_x.npy, test_y.npy).

Run prediction.py to perform inference using the saved weights.

🧮 Key Functions (from code)
def sigmoid(X, theta, B):
    return 1 / (1 + np.exp(-(np.dot(theta, X.T) + B)))

def cost(X, y, theta):
    # Regularized and unregularized cost computed
    ...

def gradient_descent(X, y, theta, B, alpha, iterations):
    # Weight updates via backpropagation
    ...

🖼️ UI & Visualization
UI built with PyQt5

CSV logging of predictions

Future enhancements could include ROC curve visualization and class distribution charts.

📌 To Run
Installation
pip install numpy pandas scikit-learn PyQt5

Training
python Logistic_regression.py

Prediction
python prediction.py

🚀 Potential Enhancements
Switch from manual cost calculation to scikit-learn or statsmodels for benchmarking.

Implement ROC-AUC and Precision-Recall metrics.

Add an LLM-based UI overlay for fraud pattern explanation.

Create a FastAPI endpoint for real-time deployment.

👨‍💻 Author
Faizan — AI/ML Engineer 🔧 Specializing in scalable, web-integrated ML/DL systems 🔗 GitHub Profile
