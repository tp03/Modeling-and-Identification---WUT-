import matplotlib.pyplot as plt
import numpy as np

# Using readlines()
file1 = open('danestat22.txt', 'r')
Lines = file1.readlines()

u_learn = []
y_learn = []

u_wer = []
y_wer = []

count = 1
for line in Lines:
    extr = line.split()
    if count % 2 != 0:
        y_learn.append(float(extr[1]))
        u_learn.append(float(extr[0]))
        count += 1
    else:
        y_wer.append(float(extr[1]))
        u_wer.append(float(extr[0]))
        count += 1


# plt.plot(u_learn, y_learn, 'ro')
# plt.show()

# plt.plot(u_wer, y_wer, 'ro')
# plt.show()

M = np.array([[1, u_learn[0]]])
Y = np.array([y_learn])
Y = Y.T


for i in range(len(y_learn) -1):
    extras = np.array([1, u_learn[i+1]])
    M = np.r_[M, [extras]]

first_m = np.matmul(M.T, M)
second_m = np.matmul(np.linalg.inv(first_m), M.T)
W = np.matmul(second_m, Y)

y_counted1  = []
for point in u_learn:
    y_counted1.append(W[0][0] + W[1][0]*point)


learn_error = 0
for i in range(len(y_counted1)):
    learn_error += (y_counted1[i] - y_learn[i])**2

plt.plot(u_learn, y_counted1, 'ro', color = 'b')
plt.plot(u_learn, y_learn, 'ro')
plt.legend(['wyjście modelu', 'dane uczące'])
plt.title(f"Model statyczny liniowy. Błąd: {learn_error}")
plt.show()

M = np.array([[1, u_wer[0]]])
Y = np.array([y_wer])
Y = Y.T


for i in range(len(y_wer) -1):
    extras = np.array([1, u_wer[i+1]])
    M = np.r_[M, [extras]]

first_m = np.matmul(M.T, M)
second_m = np.matmul(np.linalg.inv(first_m), M.T)
W = np.matmul(second_m, Y)

y_counted2  = []
for point in u_wer:
    y_counted2.append(W[0][0] + W[1][0]*point)


wer_error = 0
for i in range(len(y_counted2)):
    wer_error += (y_counted2[i] - y_wer[i])**2

plt.plot(u_wer, y_counted2, 'ro', color = 'b')
plt.plot(u_wer, y_wer, 'ro')
plt.legend(['wyjście modelu', 'dane weryfikujące'])
plt.title(f"Model statyczny liniowy. Błąd: {wer_error}")
plt.show()
