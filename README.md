# Solar-Power-Prediction
Real-time weather prediction system using multiple ML models (Random Forest, XGBoost, CatBoost, etc.) to forecast temperature, GHI, and power output. Data is stored in a MySQL database and visualized live in Grafana for continuous monitoring.

# 🌦️ Real-Time Weather Prediction and Dashboard System

This project continuously predicts **Temperature**, **Global Horizontal Irradiance (GHI)**, and **Power Output** using multiple machine learning models.  
All results are stored in a **MySQL database** and visualized in **Grafana** for real-time monitoring.

---

## ⚙️ Project Workflow

### 🧠 STEP 1: Connect to MySQL Database  
Establish a secure connection to the MySQL server for data storage.

### 🔁 STEP 2: Start Infinite Loop  
Run an endless loop to make predictions continuously in real time.

### 📂 STEP 3: Load & Preprocess Data  
Load the dataset "niteditedfinal.csv", filling missing values with zeros.

### 📊 STEP 4: Extract Features & Targets  
Separate feature columns and target columns for "Temperature" and "GHI".

### ✂️ STEP 5: Split Data  
Split the dataset into training and testing sets for both prediction targets.

### 🤖 STEP 6: Train Models  
Initialize and train the following models for both "Temperature" and "GHI":
- Random Forest  
- Linear Regression  
- Support Vector Regressor (SVR)  
- XGBoost  
- CatBoost  
- AdaBoost  
- Gradient Boost  

### ⏰ STEP 7: Generate Future Time Features  
Add "15 minutes" to the current time and format it as input for the models.

### 🔮 STEP 8: Predict Future Values  
Predict "Temperature" and "GHI" for the next 15-minute interval.

### ⚡ STEP 9: Calculate Power  
Compute power output using a formula based on predicted Temperature and GHI values.

### 🧾 STEP 10: Format Data for SQL  
Convert predictions to floats and prepare them for SQL insertion.

### 🗄️ STEP 11: Insert Predictions into Database  
Insert "time", "temperature", "GHI", and "power" values into the MySQL table.

### 💤 STEP 12: Loop Delay  
Pause for "1 second" before repeating the process.

### 🖥️ STEP 13: Setup XAMPP Server  
Install and configure "XAMPP" to host your MySQL database.

### 🧱 STEP 14: Create Table Structure  
Define a table with columns and datatypes for time, temperature, GHI, and power.

### 📊 STEP 15: Visualize in Grafana  
Connect the MySQL database to "Grafana" and create a real-time dashboard.

---

## 🛠️ Requirements
- Python 3.x  
- MySQL / XAMPP Server  
- Grafana  

  pip install pandas numpy scikit-learn xgboost catboost mysql-connector-python
