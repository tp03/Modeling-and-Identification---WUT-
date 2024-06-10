import matplotlib.pyplot as plt
import numpy as np

# Using readlines()
file1 = open('danedynucz22.txt', 'r')
Lines = file1.readlines()

file2 = open('danedynwer22.txt', 'r')
Lines2 = file2.readlines()

N = 3
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

# plt.plot(u_learn, y_learn, 'ro')
# plt.show()

# plt.plot(u_wer, y_wer, 'ro')
# plt.show()

Y = np.array([y_learn[N:]])
Y = Y.T

first_row = []
for i in range(N):
    counted_u = u_learn[N-i-1]
    first_row.append(counted_u)
for j in range(N):
    counted_y = y_learn[N-j-1]
    first_row.append(counted_y)


M = np.array([first_row])

for i in range(len(Y) - 1):
    row = []
    for j in range(N):
        counted_u = u_learn[i+N-j]
        row.append(counted_u)
    for k in range(N):
        counted_y = y_learn[i+N-k]
        row.append(counted_y)

    extras = np.array([row])
    M = np.r_[M, extras]

first_m = np.matmul(M.T, M)
second_m = np.matmul(np.linalg.inv(first_m), M.T)
W = np.matmul(second_m, Y)


if recursion is False:
    y_counted1 = np.matmul(M,W)
    
    first_row = []
    for i in range(N):
        counted_u = u_wer[N-i-1]
        first_row.append(counted_u)
    for j in range(N):
        counted_y = y_wer[N-j-1]
        first_row.append(counted_y)


    M = np.array([first_row])

    for i in range(len(Y) - 1):
        row = []
        for j in range(N):
            counted_u = u_wer[i+N-j]
            row.append(counted_u)
        for k in range(N):
            counted_y = y_wer[i+N-k]
            row.append(counted_y)

        extras = np.array([row])
        M = np.r_[M, extras]

    learn_error = 0
    wer_error = 0

    y_counted2 = np.matmul(M,W)

    for i in range(len(M)):
        learn_error += (y_counted1[i] - y_learn[N+i])**2
        wer_error += (y_counted2[i] - y_wer[N+i])**2

    plt.figure(1)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(y_counted1, color = 'b')
    axs[0].plot(y_learn[N:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane uczące'])
    plt.title(f"Model dynamiczny bez rekurencji liniowy rzędu {N}. Błąd: {round(learn_error[0], 2)}")
    axs[1].plot(u_learn)
    
    plt.figure(2)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(y_counted2, color = 'b')
    axs[0].plot(y_wer[N:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane weryfikujące'])
    plt.title(f"Model dynamiczny bez rekurencji liniowy rzędu {N}. Błąd: {round(wer_error[0], 2)}")
    axs[1].plot(u_wer)
    plt.show()

else:

    y_counted1 = np.zeros(2000)
    y_counted2 = np.zeros(2000)
    if N == 3:
        y_counted1[0] = y_learn[0]
        y_counted1[1] = y_learn[1]
        y_counted1[2] = y_learn[2]
        y_counted2[0] = y_wer[0]
        y_counted2[1] = y_wer[1]
        y_counted2[2] = y_wer[2]
    elif N == 2:
        y_counted1[0] = y_learn[0]
        y_counted1[1] = y_learn[1]
        y_counted2[0] = y_wer[0]
        y_counted2[1] = y_wer[1]
    else:
        y_counted1[0] = y_learn[0]
        y_counted2[0] = y_wer[0]

    for i in range(len(M)):
        w = 0
        if N == 3:
            w += W[0][0] * u_learn[i+N-1]
            w += W[1][0] * u_learn[i+N-2]
            w += W[2][0] * u_learn[i+N-3]
            w += W[3][0] * y_counted1[i+N-1]
            w += W[4][0] * y_counted1[i+N-2]
            w += W[5][0] * y_counted1[i+N-3]
            y_counted1[N+i] = w
        elif N == 2:
            w += W[0][0] * u_learn[i+N-1]
            w += W[1][0] * u_learn[i+N-2]
            w += W[2][0] * y_counted1[i+N-1]
            w += W[3][0] * y_counted1[i+N-2]
            y_counted1[N+i] = w
        else:
            w += W[0][0] * u_learn[i+N-1]
            w += W[1][0] * y_counted1[i+N-1]
            y_counted1[N+i] = w



    for i in range(len(M)):
        w = 0
        if N == 3:
            w += W[0][0] * u_wer[i+N-1]
            w += W[1][0] * u_wer[i+N-2]
            w += W[2][0] * u_wer[i+N-3]
            w += W[3][0] * y_counted2[i+N-1]
            w += W[4][0] * y_counted2[i+N-2]
            w += W[5][0] * y_counted2[i+N-3]
            y_counted2[N+i] = w
        elif N == 2:
            w += W[0][0] * u_wer[i+N-1]
            w += W[1][0] * u_wer[i+N-2]
            w += W[2][0] * y_counted2[i+N-1]
            w += W[3][0] * y_counted2[i+N-2]
            y_counted2[N+i] = w
        else:
            w += W[0][0] * u_wer[i+N-1]
            w += W[1][0] * y_counted2[i+N-1]
            y_counted2[N+i] = w

    learn_error = 0
    wer_error = 0

    for i in range(len(M)):
        learn_error += (y_counted1[N+i] - y_learn[N+i])**2
        wer_error += (y_counted2[N+i] - y_wer[N+i])**2

    plt.figure(1)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(y_counted1, color = 'b')
    axs[0].plot(y_learn[N:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane uczące'])
    plt.title(f"Model dynamiczny z rekurencją liniowy rzędu {N}. Błąd: {round(learn_error, 2)}")
    axs[1].plot(u_learn)
    
    plt.figure(2)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(y_counted2, color = 'b')
    axs[0].plot(y_wer[N:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane weryfikujące'])
    plt.title(f"Model dynamiczny z rekurencją liniowy rzędu {N}. Błąd: {round(wer_error, 2)}")
    axs[1].plot(u_wer)
    plt.show()


