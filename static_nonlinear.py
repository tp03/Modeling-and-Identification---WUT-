import matplotlib.pyplot as plt
import numpy as np

# Using readlines()
file1 = open('danestat22.txt', 'r')
Lines = file1.readlines()

N = 5

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

first_row = [1]

for i in range(N):
    counted_u = u_learn[0]
    for j in range(i):
        counted_u = counted_u*u_learn[0]
    first_row.append(counted_u)


    
M = np.array([first_row])
Y = np.array([y_learn])
Y = Y.T
 
for i in range(len(y_learn) -1):
    row = [1]
    for j in range(N):
        counted_u = u_learn[i+1]
        for k in range(j):
            counted_u = counted_u*u_learn[i+1]
        row.append(counted_u)

    extras = np.array([row])
    M = np.r_[M, extras]


first_m = np.matmul(M.T, M)
second_m = np.matmul(np.linalg.inv(first_m), M.T)
W = np.matmul(second_m, Y)

y_counted1 = np.matmul(M, W)
y_counted2  = []
for point in u_wer:
    w = 0
    for i in range(len(W)):
        w += W[i][0] * point**i
    y_counted2.append(w)     

learn_error = 0
wer_error = 0

for i in range(len(y_counted1)):
    learn_error += (y_counted1[i] - y_learn[i])**2

for i in range(len(y_counted2)):
    wer_error += (y_counted2[i] - y_wer[i])**2

plt.plot(u_learn, y_counted1, 'ro', color = 'b')
plt.plot(u_learn, y_learn, 'ro')
plt.legend(['wyjście modelu', 'dane uczące'])
plt.title(f"Model statyczny nieliniowy stopnia {N}. Błąd: {learn_error[0]}")
plt.show()

plt.plot(u_wer, y_counted2, 'ro', color = 'b')
plt.plot(u_wer, y_wer, 'ro')
plt.legend(['wyjście modelu', 'dane weryfikujące'])
plt.title(f"Model statyczny nieliniowy stopnia {N}. Błąd: {wer_error}")
plt.show()
