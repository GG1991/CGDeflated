
import numpy as np
import matplotlib.pyplot as plt
from cg import cg_simple
from cg import cg_pd
from cg import cg_deflated


n = 100
dx = 1.0 / (n - 1)

A = np.zeros((n, n))
x = np.zeros(n)
c = np.linspace(0, 1, n)
b = np.zeros(n)

for i in range(1, n - 1):
    A[i,[i - 1, i, i + 1]] = [ -1.0 / dx, 2.0 / dx, -1.0 / dx]
    b[i] = +1.0

A[0,0] = 1.0
A[n - 1, n - 1] = 1.0

b[0] = 0.0
b[n - 1] = 0.0

plt.title("Solution to the 1D Laplacian")
x = np.linalg.solve(A, b)
plt.plot(c, x, 'C1', label='linalg')

#print "cg_simple"
#cg_simple(A, b, x)
#plt.plot(c, x, 'C2', label='cg_simple')

#print "cg_pd"
#cg_pd(A, b, x)
#plt.plot(c, x, 'C3', label='cg_pd')

print "cg_deflated"
cg_deflated(A, b, x, 4)
#plt.plot(c, x, 'C3', label='cg_deflated')

#legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
#
#plt.show()
