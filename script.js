async function getPrediction() {
    const ticker = document.getElementById('ticker').value;
    const loadingElement = document.getElementById('loading');
    const errorElement = document.getElementById('error');
    const resultElement = document.getElementById('result');
    const actualCloseElement = document.getElementById('actual-close');
    const rfPredictionElement = document.getElementById('rf-prediction');
    const gbPredictionElement = document.getElementById('gb-prediction');
    const xgbPredictionElement = document.getElementById('xgb-prediction');
    const rfPerformanceElement = document.getElementById('rf-performance');
    const gbPerformanceElement = document.getElementById('gb-performance');
    const xgbPerformanceElement = document.getElementById('xgb-performance');

    // Clear previous results and errors
    errorElement.textContent = '';
    resultElement.style.display = 'none';

    if (!ticker) {
        errorElement.textContent = 'Please enter a ticker symbol.';
        return;
    }

    loadingElement.style.display = 'block';

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ticker })
        });

        const data = await response.json();

        loadingElement.style.display = 'none';

        if (response.ok) {
            const { predictions, model_performance } = data;

            actualCloseElement.textContent = predictions.actual_close ? predictions.actual_close : 'N/A';
            rfPredictionElement.textContent = predictions.predicted_prices['Random Forest'];
            gbPredictionElement.textContent = predictions.predicted_prices['Gradient Boosting'];
            xgbPredictionElement.textContent = predictions.predicted_prices['XGBoost'];

            rfPerformanceElement.textContent = model_performance['Random Forest'];
            gbPerformanceElement.textContent = model_performance['Gradient Boosting'];
            xgbPerformanceElement.textContent = model_performance['XGBoost'];

            resultElement.style.display = 'block';
        } else {
            errorElement.textContent = data.error || 'An error occurred.';
        }
    } catch (error) {
        loadingElement.style.display = 'none';
        errorElement.textContent = 'Failed to fetch prediction: ' + error.message;
    }
}
