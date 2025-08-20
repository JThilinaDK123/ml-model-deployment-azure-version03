from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib
import math
from datetime import timedelta
matplotlib.use('Agg')  # Use 'Agg' backend for rendering plots

app = Flask(__name__)

# Load trained model
model = load_model('lstm_model.h5')


df_crude = pd.read_csv('FPT_Damping_Scheme_Replacements.csv')
df_crude_index = df_crude.set_index('date')
data = df_crude_index

training_data_len = math.ceil(len(data) * .8)
train_data = data[:training_data_len].iloc[:,:1]
test_data = data[training_data_len:].iloc[:,:1]

dataset_train = train_data.Crude_oil_price.values
dataset_train = np.reshape(dataset_train, (-1,1))


scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(dataset_train)
dataset_test = test_data.Crude_oil_price.values
dataset_test = np.reshape(dataset_test, (-1,1))
scaled_test = scaler.fit_transform(dataset_test)

X_train = []
y_train = []
for i in range(100, len(scaled_train)):
    X_train.append(scaled_train[i-100:i, 0])
    y_train.append(scaled_train[i, 0])
    
X_test = []
y_test = []
for i in range(100, len(scaled_test)):
    X_test.append(scaled_test[i-100:i, 0])
    y_test.append(scaled_test[i, 0])


X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))

Y_test_O = scaler.inverse_transform(y_test)


static_plots = {
    'Historical Crude Oil Prices - plot': '/static/hist_plot1.png',
    'Fixed Price Threshold (Damping Scheme) - plot': '/static/damping_plot1.png',
    'Fixed Price Threshold (Mean Replacement) - plot': '/static/mean_plot1.png',
    'Fixed Price Threshold (Median Replacement) - plot': '/static/median_plot1.png',
    'Fixed Price Threshold (Threshold Scheme) - plot': '/static/threshold_plot1.png',
}


# Function to make predictions
def make_forecast(window_size):
    last_sequence = X_test[-1]
    last_sequence = np.reshape(last_sequence, (1, X_test.shape[1], 1))
    forecasts = []

    for _ in range(window_size):
        next_value = model.predict(last_sequence)
        forecasts.append(next_value[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], np.reshape(next_value, (1, 1, 1)), axis=1)

    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
    return forecasts



# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    forecast_plot = None
    selected_plot = None
    forecasts = None
    selected_plot_name = None

    if request.method == 'POST':
        if 'forecast' in request.form:
            window_size = int(request.form['window_size'])
            forecasts = make_forecast(window_size)

            plt.figure(figsize=(10, 5))
            
            test_dates = np.arange(len(Y_test_O))
            forecast_dates = np.arange(len(Y_test_O), len(Y_test_O) + len(forecasts))
            plt.plot(test_dates, Y_test_O, label='Actual Test Set Values')
            
            forecast_dates = np.arange(len(Y_test_O), len(Y_test_O) + len(forecasts))
            plt.plot(forecast_dates, forecasts, label=f'Forecast for Next {window_size} Days')
            plt.xlabel('Days')
            plt.ylabel('Crude Oil Prices')
            plt.legend()

            # Create a range of dates for the actual test set values
            test_dates = pd.to_datetime(test_data.index[-len(Y_test_O):])
            forecast_dates = pd.date_range(start=test_dates[-1] + timedelta(days=1), periods=len(forecasts), freq='D')
            all_dates = np.concatenate([test_dates, forecast_dates])
            plt.figure(figsize=(12, 6))
            plt.plot(all_dates[:len(Y_test_O)], Y_test_O, label='Actual Test Set Values')
            plt.plot(all_dates[len(Y_test_O):], forecasts, label='Forecast for Next {} Days'.format(len(forecasts)))
            plt.xlabel('Date')
            plt.ylabel('Crude Oil Prices')
            plt.legend()

            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()  # Close the figure to avoid memory leaks
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            forecast_plot = f'data:image/png;base64,{plot_url}'

        if 'static_plot' in request.form:
            # selected_plot = static_plots[request.form['static_plot']]
            selected_plot_name = request.form['static_plot']
            selected_plot = static_plots[selected_plot_name]


    return render_template('home.html', forecast_plot=forecast_plot, static_plots=static_plots, selected_plot=selected_plot,
                           forecasts=forecasts, selected_plot_name=selected_plot_name)



@app.route('/download', methods=['POST'])
def download():
    window_size = int(request.form['window_size'])
    forecasts = make_forecast(window_size)
    
    # Create a DataFrame for the forecasted data
    test_dates = pd.to_datetime(test_data.index[-len(Y_test_O):])
    forecast_dates = pd.date_range(start=test_dates[-1] + timedelta(days=1), periods=len(forecasts), freq='D')
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Price': forecasts.flatten()
    })
    
    # Save the forecasted data to a CSV file
    forecast_csv = io.BytesIO()
    forecast_df.to_csv(forecast_csv, index=False)
    forecast_csv.seek(0)
    
    return send_file(forecast_csv, as_attachment=True, download_name='forecasted_prices.csv', mimetype='text/csv')



if __name__ == '__main__':
    app.run(debug=True)