
import numpy as np
import matplotlib.pyplot as plt

n = 100

A = np.zeros((n, n))
x = np.zeros(n)
c = np.linspace(0, 1, n)
b = np.zeros(n)

for i in range(1, n - 1):
    A[i,[i - 1, i, i + 1]] = [ -1.0, 2.0, -1.0]
    b[i] = 1.0 

A[0,0] = 1.0
A[n - 1, n - 1] = 1.0

b[0] = 1.0 
b[n - 1] = 1.0 

x = np.linalg.solve(A, b)

plt.title("Solution to the 1D Laplacian")
plt.plot(c, x)
plt.show()
