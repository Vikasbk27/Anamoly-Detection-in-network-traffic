# Network Anomaly Detection 🔍

This project focuses on detecting anomalies in a network using Machine Learning techniques. It processes network data, selects key features, trains a model, and evaluates its accuracy to identify abnormal patterns in the network.

---

## 🚀 Project Overview

- **Domain:** Network Security
- **Objective:** Detect anomalies in network traffic using Logistic Regression and Recursive Feature Elimination (RFE)
- **Data:** Train and Test datasets in CSV format

---

## 📁 Project Structure


---

## ⚙️ Technologies Used

- Python
- Jupyter Notebook
- Scikit-learn
- LightGBM
- XGBoost
- Pandas, NumPy, Seaborn, Matplotlib
- Pickle (for saving model)

---

## 🧠 ML Pipeline

1. **Data Preprocessing**
   - Label encoding of categorical features
   - Feature elimination (`num_outbound_cmds`)
   - Feature selection using `RFE`

2. **Model Training**
   - Logistic Regression used as classifier
   - Features scaled using `StandardScaler`
   - Dataset split into training and testing

3. **Model Evaluation**
   - Accuracy printed for the classifier

4. **Deployment**
   - Trained model saved as `model.pkl` using Pickle

---

## 📈 Model Performance

- **Algorithm:** Logistic Regression
- **Feature Selection:** Recursive Feature Elimination (RFE)
- **Accuracy:** ~97% (based on evaluation on test data)

---

## 📦 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Network-Anomaly-Detection.git
   cd Network-Anomaly-Detection
Run the Jupyter notebook:

    jupyter notebook network.ipynb

To make predictions
   
    import pickle
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict(your_input_data)

