import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import timedelta
import pandas as pd

DAYS_BACK = 60
TRAIN_PERCENT = 0.8
X_FUTURE = 5


def retrieve_stock_data():
    data = yf.download("AAPL", start="2015-01-01", end="2023-04-24")
    return data


def extract_prices(data):
    prices = data['Close'].tolist()
    dataset = np.array(prices).reshape(-1, 1)
    return dataset


def scale_data(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    training_data_len = math.ceil(len(dataset) * TRAIN_PERCENT)
    train_data = scaled_data[0:training_data_len, :]
    return train_data, scaler, training_data_len, scaled_data


def split_data(train_data, days_back):
    x_train = []
    y_train = []
    for i in range(days_back, len(train_data)):
        x_train.append(train_data[i - days_back:i, 0])  # Will contain DAYS_BACK values (0-59) in each row
        y_train.append(train_data[i, 0])  # Will contain the 61st value (DAYS_BACK) in each row

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train


def build_lstm_model(x_train_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train_shape, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, x_train, y_train, batch_size, epochs):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model


def create_testing_data_set(scaled_data, training_data_len, days_back, dataset):
    test_data = scaled_data[training_data_len - days_back:]
    # Create the data set x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(days_back, len(test_data)):
        x_test.append(test_data[i - days_back:i, 0])
    x_test = np.array(x_test)
    print(x_test.shape)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test


def get_model_predictions(model, x_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions


def evaluate_model(predictions, y_test):
    root_mean_squared_error = np.sqrt(np.mean(predictions - y_test) ** 2)
    print("model error:", root_mean_squared_error)


def evaluate_prediction_trend(predictions, y_test):
    trend = []
    for i in range(len(predictions)):
        if i == 0:
            trend.append(-1)  # Cannot calculate for the first day
        elif predictions[i] > predictions[i - 1] and y_test[i] > y_test[i - 1]:
            trend.append(1)  # Both stock and prediction increased
        elif predictions[i] < predictions[i - 1] and y_test[i] < y_test[i - 1]:
            trend.append(1)  # Both stock and prediction decreased
        else:
            trend.append(0)  # One increased and one decreased
    print("rights: ", trend.count(1))
    print("mistakes: ", trend.count(0))


def plot_data(train, valid, predictions):
    valid['Predictions'] = predictions

    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
    plt.show()


def predict_future_stock_prices(model, x_test, scaler, x_future):
    predictions = np.array([])
    last = x_test[-1]
    for i in range(x_future):
        curr_prediction = model.predict(np.array([last]))
        print(curr_prediction)
        last = np.concatenate([last[1:], curr_prediction])
        predictions = np.concatenate([predictions, curr_prediction[0]])
    predictions = scaler.inverse_transform([predictions])[0]
    print(predictions)
    return predictions


def create_predictions_dataframe(data, predictions, x_future):
    dicts = []
    curr_date = data.index[-1]
    for i in range(x_future):
        curr_date = curr_date + timedelta(days=1)
        dicts.append({'Predictions': predictions[i], "Date": curr_date})

    return pd.DataFrame(dicts).set_index("Date")


def main():
    data = retrieve_stock_data()
    dataset = extract_prices(data)
    train_data, scaler, training_data_len, scaled_data = scale_data(dataset)

    x_train, y_train = split_data(train_data, DAYS_BACK)
    x_test, y_test = create_testing_data_set(scaled_data, training_data_len, DAYS_BACK, dataset)
    model = build_lstm_model(x_train.shape[1])
    model = train_model(model, x_train, y_train, 10, 50)
    predictions = get_model_predictions(model, x_test, scaler)

    evaluate_model(predictions, y_test)
    evaluate_prediction_trend(predictions, y_test)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    plot_data(train, valid, predictions)

    predictions = predict_future_stock_prices(model, x_test, scaler, X_FUTURE)
    new_data = create_predictions_dataframe(data, predictions, X_FUTURE)

    # Visualize the data
    train = data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(new_data['Predictions'])
    plt.legend(['Train', 'Predictions'], loc='lower right')
    plt.show()

    valid['Predictions'] = predictions


if __name__ == "__main__":
    main()
