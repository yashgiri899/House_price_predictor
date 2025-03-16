# 🏡 House Price Prediction using Random Forest

## 📌 Overview
This project implements a **Random Forest Regressor** to predict house prices based on various features such as square footage, number of bedrooms, lot size, and neighborhood quality. The model undergoes **feature engineering, preprocessing, hyperparameter tuning, and evaluation** to ensure optimal performance.

## 🚀 Features
- **Feature Engineering**: Converts `Year_Built` into `House_Age` and categorizes `Neighborhood_Quality`.
- **Data Preprocessing**: One-hot encoding, scaling numerical features.
- **Model Training**: Utilizes `RandomForestRegressor` with optimized hyperparameters.
- **Evaluation Metrics**: Mean Squared Error (MSE) and R² score.
- **Cross-Validation**: Ensures model stability across different subsets of data.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to find the best parameters.
- **Feature Importance Analysis**: Identifies the most significant predictors.
- **Model Saving**: Saves the trained model and scaler for future use.

---

## 📂 Project Structure
```
├── house_price_rf.py      # Main Python script for training and evaluation
├── house_price_regression_dataset.csv # Sample dataset (not included in repo)
├── house_price_model.pkl  # Saved model after training (generated)
├── README.md              # Project documentation
```

---

## 📊 Dataset
The dataset should contain the following features:
- `Square_Footage` (numeric)
- `Num_Bedrooms` (numeric)
- `Num_Bathrooms` (numeric)
- `Lot_Size` (numeric)
- `Garage_Size` (numeric)
- `Year_Built` (numeric)
- `Neighborhood_Quality` (scale 1-10)
- `House_Price` (target variable, numeric)

---

## 🛠️ Installation & Usage
### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```
### 2️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```
### 3️⃣ **Run the Model**
```sh
python house_price_rf.py house_price_regression_dataset.csv
```

---

## 📈 Model Performance
- **Mean Squared Error (MSE)**: 377,815,577.02
- **R² Score**: ~0.99
- **Cross-Validation R² Scores**: `[0.9928, 0.9923, 0.9943, 0.9931, 0.9929]`
- **Best Hyperparameters**:
  ```json
  {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2
  }
  ```

---

## 🤖 Future Improvements
- Implement **XGBoost or LightGBM** for comparison.
- Deploy the model as an **API (FastAPI/Flask)**.
- Add **automated feature selection**.
- Use **SHAP values** for better interpretability.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 👨‍💻 Author
- **Your Name**  
- 📧 your.email@example.com  
- 🌐 [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourname)

---

⭐ **Feel free to contribute and star this repository if you found it helpful!** ⭐
