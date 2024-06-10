import matplotlib.pyplot as plt
import numpy as np

# Using readlines()
file1 = open('danedynucz22.txt', 'r')
Lines = file1.readlines()

file2 = open('danedynwer22.txt', 'r')
Lines2 = file2.readlines()

N = 4
D = 3
recursion = True
get_static = True

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

Y = np.array([y_learn[D:]])
Y = Y.T

first_row = []
for i in range(D):
    for j in range(N):
        counted_u = u_learn[D-i-1]
        for k in range(j):
            counted_u = counted_u*u_learn[D-i-1]
        first_row.append(counted_u)
    for j in range(N):
        counted_y = y_learn[D-i-1]
        for k in range(j):
            counted_y = counted_y*y_learn[D-i-1]
        first_row.append(counted_y)


M = np.array([first_row])

for i in range(len(Y) - 1):
    row = []
    for m in range(D):
        for j in range(N):
            counted_u = u_learn[i+D-m]
            for k in range(j):
                counted_u = counted_u*u_learn[i+D-m]
            row.append(counted_u)
        for j in range(N):
            counted_y = y_learn[i+D-m]
            for k in range(j):
                counted_y = counted_y*y_learn[i+D-m]
            row.append(counted_y)

    extras = np.array([row])
    M = np.r_[M, extras]

first_m = np.matmul(M.T, M)
second_m = np.matmul(np.linalg.inv(first_m), M.T)
W = np.matmul(second_m, Y)


if recursion is False:
    y_counted1 = np.matmul(M, W)
    
    first_row = []
    for i in range(D):
        for j in range(N):
            counted_u = u_wer[D-i-1]
            for k in range(j):
                counted_u = counted_u*u_wer[D-i-1]
            first_row.append(counted_u)
        for j in range(N):
            counted_y = y_wer[D-i-1]
            for k in range(j):
                counted_y = counted_y*y_wer[D-i-1]
            first_row.append(counted_y)


    M = np.array([first_row])

    for i in range(len(Y) - 1):
        row = []
        for m in range(D):
            for j in range(N):
                counted_u = u_wer[i+D-m]
                for k in range(j):
                    counted_u = counted_u*u_wer[i+D-m]
                row.append(counted_u)
            for j in range(N):
                counted_y = y_wer[i+D-m]
                for k in range(j):
                    counted_y = counted_y*y_wer[i+D-m]
                row.append(counted_y)

        extras = np.array([row])
        M = np.r_[M, extras]
        learn_error = 0
        wer_error = 0

    y_counted2 = np.matmul(M, W)

    for i in range(len(M)):
        learn_error += (y_counted1[i] - y_learn[D+i])**2
        wer_error += (y_counted2[i] - y_wer[D+i])**2

    plt.figure(1)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(y_counted1, color = 'b')
    axs[0].plot(y_learn[N:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane uczące'])
    plt.title(f"Model dynamiczny bez rekurencji nieliniowy rzędu {D} stopnia {N}. Błąd: {round(learn_error[0], 2)}")
    axs[1].plot(u_learn)
    
    plt.figure(2)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(y_counted2, color = 'b')
    axs[0].plot(y_wer[N:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane weryfikujące'])
    plt.title(f"Model dynamiczny bez rekurencji nieliniowy rzędu {D} stopnia {N}. Błąd: {round(wer_error[0], 2)}")
    axs[1].plot(u_wer)
    plt.show()

else:

    y_counted1 = np.zeros(2000)
    y_counted2 = np.zeros(2000)
    if D == 3:
        y_counted1[0] = y_learn[0]
        y_counted1[1] = y_learn[1]
        y_counted1[2] = y_learn[2]
        y_counted2[0] = y_wer[0]
        y_counted2[1] = y_wer[1]
        y_counted2[2] = y_wer[2]
    elif D == 2:
        y_counted1[0] = y_learn[0]
        y_counted1[1] = y_learn[1]
        y_counted2[0] = y_wer[0]
        y_counted2[1] = y_wer[1]
    else:
        y_counted1[0] = y_learn[0]
        y_counted2[0] = y_wer[0]

    for i in range(len(M)):
        w = 0
        if D == 3:
            if N == 1:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * y_counted1[i+D-1]
                w += W[2][0] * u_learn[i+D-2]
                w += W[3][0] * y_counted1[i+D-2]
                w += W[4][0] * u_learn[i+D-3]
                w += W[5][0] * y_counted1[i+D-3]
                y_counted1[D+i] = w

            elif N == 2:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * y_counted1[i+D-1]
                w += W[3][0] * y_counted1[i+D-1]**2
                w += W[4][0] * u_learn[i+D-2]
                w += W[5][0] * u_learn[i+D-2]**2
                w += W[6][0] * y_counted1[i+D-2]
                w += W[7][0] * y_counted1[i+D-2]**2
                w += W[8][0] * u_learn[i+D-3]
                w += W[9][0] * u_learn[i+D-3]**2
                w += W[10][0] * y_counted1[i+D-3]
                w += W[11][0] * y_counted1[i+D-3]**2
                y_counted1[D+i] = w
            elif N == 3:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * y_counted1[i+D-1]
                w += W[4][0] * y_counted1[i+D-1]**2
                w += W[5][0] * y_counted1[i+D-1]**3
                w += W[6][0] * u_learn[i+D-2]
                w += W[7][0] * u_learn[i+D-2]**2
                w += W[8][0] * u_learn[i+D-2]**3
                w += W[9][0] * y_counted1[i+D-2]
                w += W[10][0] * y_counted1[i+D-2]**2
                w += W[11][0] * y_counted1[i+D-2]**3
                w += W[12][0] * u_learn[i+D-3]
                w += W[13][0] * u_learn[i+D-3]**2
                w += W[14][0] * u_learn[i+D-3]**3
                w += W[15][0] * y_counted1[i+D-3]
                w += W[16][0] * y_counted1[i+D-3]**2
                w += W[17][0] * y_counted1[i+D-3]**3
                y_counted1[D+i] = w
            elif N == 4:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * u_learn[i+D-1]**4
                w += W[4][0] * y_counted1[i+D-1]
                w += W[5][0] * y_counted1[i+D-1]**2
                w += W[6][0] * y_counted1[i+D-1]**3
                w += W[7][0] * y_counted1[i+D-1]**4
                w += W[8][0] * u_learn[i+D-2]
                w += W[9][0] * u_learn[i+D-2]**2
                w += W[10][0] * u_learn[i+D-2]**3
                w += W[11][0] * u_learn[i+D-2]**4
                w += W[12][0] * y_counted1[i+D-2]
                w += W[13][0] * y_counted1[i+D-2]**2
                w += W[14][0] * y_counted1[i+D-2]**3
                w += W[15][0] * y_counted1[i+D-2]**4
                w += W[16][0] * u_learn[i+D-3]
                w += W[17][0] * u_learn[i+D-3]**2
                w += W[18][0] * u_learn[i+D-3]**3
                w += W[19][0] * u_learn[i+D-3]**4
                w += W[20][0] * y_counted1[i+D-3]
                w += W[21][0] * y_counted1[i+D-3]**2
                w += W[22][0] * y_counted1[i+D-3]**3
                w += W[23][0] * y_counted1[i+D-3]**4
                y_counted1[D+i] = w
            else:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * u_learn[i+D-1]**4
                w += W[4][0] * u_learn[i+D-1]**5
                w += W[5][0] * y_counted1[i+D-1]
                w += W[6][0] * y_counted1[i+D-1]**2
                w += W[7][0] * y_counted1[i+D-1]**3
                w += W[8][0] * y_counted1[i+D-1]**4
                w += W[9][0] * y_counted1[i+D-1]**5
                w += W[10][0] * u_learn[i+D-2]
                w += W[11][0] * u_learn[i+D-2]**2
                w += W[12][0] * u_learn[i+D-2]**3
                w += W[13][0] * u_learn[i+D-2]**4
                w += W[14][0] * u_learn[i+D-2]**5
                w += W[15][0] * y_counted1[i+D-2]
                w += W[16][0] * y_counted1[i+D-2]**2
                w += W[17][0] * y_counted1[i+D-2]**3
                w += W[18][0] * y_counted1[i+D-2]**4
                w += W[19][0] * y_counted1[i+D-2]**5
                w += W[20][0] * u_learn[i+D-3]
                w += W[21][0] * u_learn[i+D-3]**2
                w += W[22][0] * u_learn[i+D-3]**3
                w += W[23][0] * u_learn[i+D-3]**4
                w += W[24][0] * u_learn[i+D-3]**5
                w += W[25][0] * y_counted1[i+D-3]
                w += W[26][0] * y_counted1[i+D-3]**2
                w += W[27][0] * y_counted1[i+D-3]**3
                w += W[28][0] * y_counted1[i+D-3]**4
                w += W[29][0] * y_counted1[i+D-3]**5
                y_counted1[D+i] = w
        elif D == 2:
            if N == 1:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * y_counted1[i+D-1]
                w += W[2][0] * u_learn[i+D-2]
                w += W[3][0] * y_counted1[i+D-2]
                y_counted1[D+i] = w

            elif N == 2:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * y_counted1[i+D-1]
                w += W[3][0] * y_counted1[i+D-1]**2
                w += W[4][0] * u_learn[i+D-2]
                w += W[5][0] * u_learn[i+D-2]**2
                w += W[6][0] * y_counted1[i+D-2]
                w += W[7][0] * y_counted1[i+D-2]**2
                y_counted1[D+i] = w

            elif N == 3:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * y_counted1[i+D-1]
                w += W[4][0] * y_counted1[i+D-1]**2
                w += W[5][0] * y_counted1[i+D-1]**3
                w += W[6][0] * u_learn[i+D-2]
                w += W[7][0] * u_learn[i+D-2]**2
                w += W[8][0] * u_learn[i+D-2]**3
                w += W[9][0] * y_counted1[i+D-2]
                w += W[10][0] * y_counted1[i+D-2]**2
                w += W[11][0] * y_counted1[i+D-2]**3
                y_counted1[D+i] = w 
            elif N == 4:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * u_learn[i+D-1]**4
                w += W[4][0] * y_counted1[i+D-1]
                w += W[5][0] * y_counted1[i+D-1]**2
                w += W[6][0] * y_counted1[i+D-1]**3
                w += W[7][0] * y_counted1[i+D-1]**4
                w += W[8][0] * u_learn[i+D-2]
                w += W[9][0] * u_learn[i+D-2]**2
                w += W[10][0] * u_learn[i+D-2]**3
                w += W[11][0] * u_learn[i+D-2]**4
                w += W[12][0] * y_counted1[i+D-2]
                w += W[13][0] * y_counted1[i+D-2]**2
                w += W[14][0] * y_counted1[i+D-2]**3
                w += W[15][0] * y_counted1[i+D-2]**4
                y_counted1[D+i] = w
            else:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * u_learn[i+D-1]**4
                w += W[4][0] * u_learn[i+D-1]**5
                w += W[5][0] * y_counted1[i+D-1]
                w += W[6][0] * y_counted1[i+D-1]**2
                w += W[7][0] * y_counted1[i+D-1]**3
                w += W[8][0] * y_counted1[i+D-1]**4
                w += W[9][0] * y_counted1[i+D-1]**5
                w += W[10][0] * u_learn[i+D-2]
                w += W[11][0] * u_learn[i+D-2]**2
                w += W[12][0] * u_learn[i+D-2]**3
                w += W[13][0] * u_learn[i+D-2]**4
                w += W[14][0] * u_learn[i+D-2]**5
                w += W[15][0] * y_counted1[i+D-2]
                w += W[16][0] * y_counted1[i+D-2]**2
                w += W[17][0] * y_counted1[i+D-2]**3
                w += W[18][0] * y_counted1[i+D-2]**4
                w += W[19][0] * y_counted1[i+D-2]**5
                y_counted1[D+i] = w
        else:
            if N == 1:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * y_counted1[i+D-1]
                y_counted1[D+i] = w

            elif N == 2:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * y_counted1[i+D-1]
                w += W[3][0] * y_counted1[i+D-1]**2
                y_counted1[D+i] = w

            elif N == 3:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * y_counted1[i+D-1]
                w += W[4][0] * y_counted1[i+D-1]**2
                w += W[5][0] * y_counted1[i+D-1]**3
                y_counted1[D+i] = w
            elif N == 4:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * u_learn[i+D-1]**4
                w += W[4][0] * y_counted1[i+D-1]
                w += W[5][0] * y_counted1[i+D-1]**2
                w += W[6][0] * y_counted1[i+D-1]**3 
                w += W[7][0] * y_counted1[i+D-1]**4
                y_counted1[D+i] = w
            else:
                w += W[0][0] * u_learn[i+D-1]
                w += W[1][0] * u_learn[i+D-1]**2
                w += W[2][0] * u_learn[i+D-1]**3
                w += W[3][0] * u_learn[i+D-1]**4
                w += W[4][0] * u_learn[i+D-1]**5
                w += W[5][0] * y_counted1[i+D-1]
                w += W[6][0] * y_counted1[i+D-1]**2
                w += W[7][0] * y_counted1[i+D-1]**3 
                w += W[8][0] * y_counted1[i+D-1]**4
                w += W[9][0] * y_counted1[i+D-1]**5
                y_counted1[D+i] = w

    for i in range(len(M)):
        w = 0
        if D == 3:
            if N == 1:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * y_counted2[i+D-1]
                w += W[2][0] * u_wer[i+D-2]
                w += W[3][0] * y_counted2[i+D-2]
                w += W[4][0] * u_wer[i+D-3]
                w += W[5][0] * y_counted2[i+D-3]
                y_counted2[D+i] = w

            elif N == 2:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * y_counted2[i+D-1]
                w += W[3][0] * y_counted2[i+D-1]**2
                w += W[4][0] * u_wer[i+D-2]
                w += W[5][0] * u_wer[i+D-2]**2
                w += W[6][0] * y_counted2[i+D-2]
                w += W[7][0] * y_counted2[i+D-2]**2
                w += W[8][0] * u_wer[i+D-3]
                w += W[9][0] * u_wer[i+D-3]**2
                w += W[10][0] * y_counted2[i+D-3]
                w += W[11][0] * y_counted2[i+D-3]**2
                y_counted2[D+i] = w
            elif N == 3:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * y_counted2[i+D-1]
                w += W[4][0] * y_counted2[i+D-1]**2
                w += W[5][0] * y_counted2[i+D-1]**3
                w += W[6][0] * u_wer[i+D-2]
                w += W[7][0] * u_wer[i+D-2]**2
                w += W[8][0] * u_wer[i+D-2]**3
                w += W[9][0] * y_counted2[i+D-2]
                w += W[10][0] * y_counted2[i+D-2]**2
                w += W[11][0] * y_counted2[i+D-2]**3
                w += W[12][0] * u_wer[i+D-3]
                w += W[13][0] * u_wer[i+D-3]**2
                w += W[14][0] * u_wer[i+D-3]**3
                w += W[15][0] * y_counted2[i+D-3]
                w += W[16][0] * y_counted2[i+D-3]**2
                w += W[17][0] * y_counted2[i+D-3]**3
                y_counted2[D+i] = w
            elif N == 4:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * u_wer[i+D-1]**4
                w += W[4][0] * y_counted2[i+D-1]
                w += W[5][0] * y_counted2[i+D-1]**2
                w += W[6][0] * y_counted2[i+D-1]**3
                w += W[7][0] * y_counted2[i+D-1]**4
                w += W[8][0] * u_wer[i+D-2]
                w += W[9][0] * u_wer[i+D-2]**2
                w += W[10][0] * u_wer[i+D-2]**3
                w += W[11][0] * u_wer[i+D-2]**4
                w += W[12][0] * y_counted2[i+D-2]
                w += W[13][0] * y_counted2[i+D-2]**2
                w += W[14][0] * y_counted2[i+D-2]**3
                w += W[15][0] * y_counted2[i+D-2]**4
                w += W[16][0] * u_wer[i+D-3]
                w += W[17][0] * u_wer[i+D-3]**2
                w += W[18][0] * u_wer[i+D-3]**3
                w += W[19][0] * u_wer[i+D-3]**4
                w += W[20][0] * y_counted2[i+D-3]
                w += W[21][0] * y_counted2[i+D-3]**2
                w += W[22][0] * y_counted2[i+D-3]**3
                w += W[23][0] * y_counted2[i+D-3]**4
                y_counted2[D+i] = w
            else:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * u_wer[i+D-1]**4
                w += W[4][0] * u_wer[i+D-1]**5
                w += W[5][0] * y_counted2[i+D-1]
                w += W[6][0] * y_counted2[i+D-1]**2
                w += W[7][0] * y_counted2[i+D-1]**3
                w += W[8][0] * y_counted2[i+D-1]**4
                w += W[9][0] * y_counted2[i+D-1]**5
                w += W[10][0] * u_wer[i+D-2]
                w += W[11][0] * u_wer[i+D-2]**2
                w += W[12][0] * u_wer[i+D-2]**3
                w += W[13][0] * u_wer[i+D-2]**4
                w += W[14][0] * u_wer[i+D-2]**5
                w += W[15][0] * y_counted2[i+D-2]
                w += W[16][0] * y_counted2[i+D-2]**2
                w += W[17][0] * y_counted2[i+D-2]**3
                w += W[18][0] * y_counted2[i+D-2]**4
                w += W[19][0] * y_counted2[i+D-2]**5
                w += W[20][0] * u_wer[i+D-3]
                w += W[21][0] * u_wer[i+D-3]**2
                w += W[22][0] * u_wer[i+D-3]**3
                w += W[23][0] * u_wer[i+D-3]**4
                w += W[24][0] * u_wer[i+D-3]**5
                w += W[25][0] * y_counted2[i+D-3]
                w += W[26][0] * y_counted2[i+D-3]**2
                w += W[27][0] * y_counted2[i+D-3]**3
                w += W[28][0] * y_counted2[i+D-3]**4
                w += W[29][0] * y_counted2[i+D-3]**5
                y_counted2[D+i] = w
        elif D == 2:
            if N == 1:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * y_counted2[i+D-1]
                w += W[2][0] * u_wer[i+D-2]
                w += W[3][0] * y_counted2[i+D-2]
                y_counted2[D+i] = w

            elif N == 2:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * y_counted2[i+D-1]
                w += W[3][0] * y_counted2[i+D-1]**2
                w += W[4][0] * u_wer[i+D-2]
                w += W[5][0] * u_wer[i+D-2]**2
                w += W[6][0] * y_counted2[i+D-2]
                w += W[7][0] * y_counted2[i+D-2]**2
                y_counted2[D+i] = w

            elif N == 3:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * y_counted2[i+D-1]
                w += W[4][0] * y_counted2[i+D-1]**2
                w += W[5][0] * y_counted2[i+D-1]**3
                w += W[6][0] * u_wer[i+D-2]
                w += W[7][0] * u_wer[i+D-2]**2
                w += W[8][0] * u_wer[i+D-2]**3
                w += W[9][0] * y_counted2[i+D-2]
                w += W[10][0] * y_counted2[i+D-2]**2
                w += W[11][0] * y_counted2[i+D-2]**3
                y_counted2[D+i] = w   

            elif N == 4:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * u_wer[i+D-1]**4
                w += W[4][0] * y_counted2[i+D-1]
                w += W[5][0] * y_counted2[i+D-1]**2
                w += W[6][0] * y_counted2[i+D-1]**3
                w += W[7][0] * y_counted2[i+D-1]**4
                w += W[8][0] * u_wer[i+D-2]
                w += W[9][0] * u_wer[i+D-2]**2
                w += W[10][0] * u_wer[i+D-2]**3
                w += W[11][0] * u_wer[i+D-2]**4
                w += W[12][0] * y_counted2[i+D-2]
                w += W[13][0] * y_counted2[i+D-2]**2
                w += W[14][0] * y_counted2[i+D-2]**3 
                w += W[15][0] * y_counted2[i+D-2]**4
                y_counted2[D+i] = w
            else:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * u_wer[i+D-1]**4
                w += W[4][0] * u_wer[i+D-1]**5
                w += W[5][0] * y_counted2[i+D-1]
                w += W[6][0] * y_counted2[i+D-1]**2
                w += W[7][0] * y_counted2[i+D-1]**3
                w += W[8][0] * y_counted2[i+D-1]**4
                w += W[9][0] * y_counted2[i+D-1]**5
                w += W[10][0] * u_wer[i+D-2]
                w += W[11][0] * u_wer[i+D-2]**2
                w += W[12][0] * u_wer[i+D-2]**3
                w += W[13][0] * u_wer[i+D-2]**4
                w += W[14][0] * u_wer[i+D-2]**5
                w += W[15][0] * y_counted2[i+D-2]
                w += W[16][0] * y_counted2[i+D-2]**2
                w += W[17][0] * y_counted2[i+D-2]**3
                w += W[18][0] * y_counted2[i+D-2]**4
                w += W[19][0] * y_counted2[i+D-2]**5
                y_counted2[D+i] = w
        else:
            if N == 1:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * y_counted2[i+D-1]
                y_counted2[D+i] = w

            elif N == 2:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * y_counted2[i+D-1]
                w += W[3][0] * y_counted2[i+D-1]**2
                y_counted2[D+i] = w

            elif N == 3:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * y_counted2[i+D-1]
                w += W[4][0] * y_counted2[i+D-1]**2
                w += W[5][0] * y_counted2[i+D-1]**3
                y_counted2[D+i] = w
            
            elif N == 4:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * u_wer[i+D-1]**4
                w += W[4][0] * y_counted2[i+D-1]
                w += W[5][0] * y_counted2[i+D-1]**2
                w += W[6][0] * y_counted2[i+D-1]**3
                w += W[7][0] * y_counted2[i+D-1]**4
                y_counted2[D+i] = w
            else:
                w += W[0][0] * u_wer[i+D-1]
                w += W[1][0] * u_wer[i+D-1]**2
                w += W[2][0] * u_wer[i+D-1]**3
                w += W[3][0] * u_wer[i+D-1]**4
                w += W[4][0] * u_wer[i+D-1]**5
                w += W[5][0] * y_counted2[i+D-1]
                w += W[6][0] * y_counted2[i+D-1]**2
                w += W[7][0] * y_counted2[i+D-1]**3 
                w += W[8][0] * y_counted2[i+D-1]**4
                w += W[9][0] * y_counted2[i+D-1]**5
                y_counted2[D+i] = w

    learn_error = 0
    wer_error = 0

    for i in range(len(M)):
        learn_error += (y_counted1[D+i] - y_learn[D+i])**2
        wer_error += (y_counted2[D+i] - y_wer[D+i])**2


    fig, axs = plt.subplots(2, 1)
    axs[0].plot(y_counted1, color = 'b')
    axs[0].plot(y_learn[N:], color = 'r')
    axs[0].legend(['wyjście modelu', 'dane uczące'])
    plt.title(f"Model dynamiczny z rekurencją nieliniowy rzędu {D} stopnia {N}. Błąd: {round(learn_error, 2)}")
    axs[1].plot(u_learn)
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(y_counted2, color = 'b')
    axs[0].plot(y_wer[N:], color = 'r')
    #axs[0].legend(['wyjście modelu', 'dane weryfikujące'])
    plt.title(f"Model dynamiczny z rekurencją nieliniowy rzędu {D} stopnia {N}. Błąd: {round(wer_error, 2)}")
    axs[1].plot(u_wer)

    if not get_static:
        plt.show()
    
#identyfikacja modelu statycznego


if get_static:
        
    b1 = W[0][0]
    b2 = W[1][0]
    b3 = W[2][0]
    b4 = W[3][0]
    b5 = W[8][0]
    b6 = W[9][0]
    b7 = W[10][0]
    b8 = W[11][0]
    b9 = W[16][0]
    b10 = W[17][0]
    b11 = W[18][0]
    b12 = W[19][0]

    a1 = W[4][0]
    a2 = W[5][0]
    a3 = W[6][0]
    a4 = W[7][0]
    a5 = W[12][0]
    a6 = W[13][0]
    a7 = W[14][0]
    a8 = W[15][0]
    a9 = W[20][0]
    a10 = W[21][0]
    a11 = W[22][0]
    a12 = W[23][0]

    u_prime = []
    y_prime = []

    for i in range(101):
        u = []
        y = []
        u_prime.append(-1+i*0.02)
        for j in range(101):
            u.append(-1+i*0.02)
            y.append(0)
        u[0] = 0
        u[1] = 0
        u[2] = 0  
        for j in range(3, 101):
            w = 0
            w += W[0][0] * u[j-1]
            w += W[1][0] * u[j-1]**2
            w += W[2][0] * u[j-1]**3
            w += W[3][0] * u[j-1]**4
            w += W[4][0] * y[j-1]
            w += W[5][0] * y[j-1]**2
            w += W[6][0] * y[j-1]**3
            w += W[7][0] * y[j-1]**4
            w += W[8][0] * u[j-2]
            w += W[9][0] * u[j-2]**2
            w += W[10][0] * u[j-2]**3
            w += W[11][0] * u[j-2]**4
            w += W[12][0] * y[j-2]
            w += W[13][0] * y[j-2]**2
            w += W[14][0] * y[j-2]**3
            w += W[15][0] * y[j-2]**4
            w += W[16][0] * u[j-3]
            w += W[17][0] * u[j-3]**2
            w += W[18][0] * u[j-3]**3
            w += W[19][0] * u[j-3]**4
            w += W[20][0] * y[j-3]
            w += W[21][0] * y[j-3]**2
            w += W[22][0] * y[j-3]**3
            w += W[23][0] * y[j-3]**4
            y[j] = w
            
        y_prime.append(y[-1])

    file1 = open('danestat22.txt', 'r')
    Lines = file1.readlines()

    u_learn = []
    y_learn = []

    for line in Lines:
        extr = line.split()
        y_learn.append(float(extr[1]))
        u_learn.append(float(extr[0]))

    plt.figure(3)
    plt.plot(u_prime, y_prime, color = 'b')
    plt.plot(u_learn, y_learn, 'ro', color = 'r')
    plt.show()