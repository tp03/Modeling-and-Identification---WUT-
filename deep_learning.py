import numpy as np
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
import matplotlib.pyplot as plt

# Wczytanie danych dynamicznych
file1 = open('danedynucz22.txt', 'r')
Lines = file1.readlines()

file2 = open('danedynwer22.txt', 'r')
Lines2 = file2.readlines()

recursion = True

u_learn = []
y_learn = []

u_wer = []
y_wer = []

for line in Lines:
    extr = line.split()
    y_learn.append(float(extr[1]))
    u_learn.append(float(extr[0]))

for line in Lines2:
    extr = line.split()
    y_wer.append(float(extr[1]))
    u_wer.append(float(extr[0]))

# Funkcja do przygotowania danych z opóźnieniami
def prepare_data(u, y, n_past):
    X, Y = [], []
    for i in range(n_past, len(u)):
        X.append([u[i - 1], u[i - 2], u[i - 3], y[i - 1], y[i - 2], y[i - 3]])
        Y.append(y[i])
    return np.array(X), np.array(Y)

# Ustawienie rzędu dynamiki
n_past = 3

# Przygotowanie danych
X_train, Y_train = prepare_data(u_learn, y_learn, n_past)
X_wer, Y_wer = prepare_data(u_wer, y_wer, n_past)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_wer = X_wer.reshape((X_wer.shape[0], 1, X_wer.shape[1]))


if not recursion:
    
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(1, X_train.shape[2])))
    model.add(Dense(1))

    model.summary()

    # Kompilacja modelu
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Trenowanie modelu
    history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_wer, Y_wer), batch_size=1, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_wer)

    learn_error = 0
    wer_error = 0

    for i in range(len(train_predict)):
            learn_error += (train_predict[i] - y_learn[3+i])**2
            wer_error += (test_predict[i] - y_wer[3+i])**2


    fig, axs = plt.subplots(2, 1)
    axs[0].plot(train_predict[:,0], color = 'b')
    axs[0].plot(y_learn[3:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane uczące'])
    plt.title(f"Błąd: {round(learn_error[0][0], 2)}")
    axs[1].plot(u_learn)
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(test_predict[:,0], color = 'b')
    axs[0].plot(y_wer[3:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane weryfikujące'])
    plt.title(f"Błąd: {round(wer_error[0][0], 2)}")
    axs[1].plot(u_wer)

    plt.show()
else:
    # Definicja modelu LSTM
    model = Sequential()
    model.add(LSTM(30, activation='tanh', input_shape=(1, X_train.shape[2])))
    model.add(Dense(1))

    model.summary()

    # Kompilacja modelu
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Trenowanie modelu
    history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_wer, Y_wer), batch_size=8, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_wer)

    learn_error = 0
    wer_error = 0

    for i in range(len(train_predict)):
            learn_error += (train_predict[i] - y_learn[3+i])**2
            wer_error += (test_predict[i] - y_wer[3+i])**2

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(train_predict[:,0], color = 'b')
    axs[0].plot(y_learn[3:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane uczące'])
    plt.title(f"Błąd: {round(learn_error[0], 2)}")
    axs[1].plot(u_learn)
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(test_predict[:,0], color = 'b')
    axs[0].plot(y_wer[3:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane weryfikujące'])
    plt.title(f"Błąd: {round(wer_error[0], 2)}")
    axs[1].plot(u_wer)

    plt.show()