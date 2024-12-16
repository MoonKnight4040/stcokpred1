import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import traceback


# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Step 1: Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:  # Check if no data is returned
        raise ValueError(f"No data found for ticker symbol: {ticker} in the specified date range.")
    stock_data.dropna(inplace=True)
    return stock_data

# Step 2: Generate features
def generate_features(data):
    data['Target'] = data['Close'].shift(-1)  # Predict next timestep Close price
    data.dropna(inplace=True)
    return data

# Step 3: Prepare data for modeling
def prepare_data(data):
    features = ['Open', 'Close', 'Volume']
    X = data[features]
    y = data['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train ensemble regressors and predict next day
def train_and_predict_next_day(X_train, X_test, y_train, y_test, recent_data):
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    next_day_predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        results[name] = round(r2, 2)  # Round model performance (R²) to 2 decimal places

        # Predict the next day
        features = ['Open', 'Close', 'Volume']
        next_features = recent_data[features].iloc[-1].values.reshape(1, -1)
        next_day_predictions[name] = round(model.predict(next_features)[0], 2)  # Round prediction to 2 decimal places

    return results, next_day_predictions

# API endpoint to fetch model performance and predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker')

    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)

    try:
        # Fetch stock data and check if it's available
        stock_data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        stock_data = generate_features(stock_data)
        X_train, X_test, y_train, y_test = prepare_data(stock_data)

        results, next_day_predictions = train_and_predict_next_day(X_train, X_test, y_train, y_test, stock_data)

        # Fetch actual close price for comparison
        next_day_data = yf.download(
            ticker, 
            start=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=2)).strftime('%Y-%m-%d')
        )

        if not next_day_data.empty:
            actual_close = next_day_data['Close'].iloc[0]
            if isinstance(actual_close, pd.Series):  # Safeguard for unexpected structure
                actual_close = actual_close.iloc[0]  # Extract scalar value

            # Ensure actual close is converted to a float (not numpy.float32) and round it
            actual_close = round(float(actual_close), 2)

            # Convert predictions to native Python floats (already rounded)
            next_day_predictions = {k: float(v) for k, v in next_day_predictions.items()}

            # Creating response structure with the actual close close to predicted prices
            response = {
                "predictions": {
                    "actual_close": actual_close,
                    "predicted_prices": next_day_predictions
                },
                "model_performance": {k: float(v) for k, v in results.items()}  # Convert model results too
            }
        else:
            response = {
                "predictions": {
                    "actual_close": None,
                    "predicted_prices": next_day_predictions,
                    "note": "No actual close price data available for the next day."
                },
                "model_performance": {k: float(v) for k, v in results.items()}
            }

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e)}), 404  # Return a 404 when ticker data is not found

    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        logging.error("Stack trace: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
