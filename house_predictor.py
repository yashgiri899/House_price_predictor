import pandas as pd
import joblib
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data(file_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """Perform feature engineering and preprocessing."""
    current_year = 2025  # Update based on current year
    df["House_Age"] = current_year - df["Year_Built"]
    df.drop(columns=["Year_Built"], inplace=True)

    # Categorizing Neighborhood_Quality
    bins = [0, 3, 7, 10]
    labels = ["Low", "Medium", "High"]
    df["Neighborhood_Category"] = pd.cut(df["Neighborhood_Quality"], bins=bins, labels=labels)
    df.drop(columns=["Neighborhood_Quality"], inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["Neighborhood_Category"], drop_first=True)
    return df


def split_and_scale_data(df, target):
    """Split dataset into training and testing sets, then scale numerical features."""
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    numerical_features = ["Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Lot_Size", "Garage_Size", "House_Age"]
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}\n")
    return mse, r2


def feature_importance(model, X):
    """Display feature importance."""
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("📌 Feature Importance:")
    print(importance, "\n")


def cross_validate(model, X, y):
    """Perform cross-validation."""
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"✅ Cross-Validation R² Scores: {scores}")
    print(f"Mean R² Score: {scores.mean():.2f}\n")


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"🏆 Best Parameters: {grid_search.best_params_}")
    print(f"Best R² Score from GridSearch: {grid_search.best_score_:.2f}\n")
    return grid_search.best_estimator_


def save_model(model, scaler, filename="house_price_model.pkl"):
    """Save trained model and scaler."""
    joblib.dump({"model": model, "scaler": scaler}, filename)
    print(f"✅ Model saved as {filename}\n")


def main(file_path):
    df = load_data(file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df, target="House_Price")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    feature_importance(model, X_train)
    cross_validate(model, X_train, y_train)
    best_model = hyperparameter_tuning(X_train, y_train)
    save_model(best_model, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to dataset CSV file")
    args = parser.parse_args()
    main(args.file_path)
