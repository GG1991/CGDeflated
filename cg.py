#
#
#
#
#

import numpy as np
import matplotlib.pyplot as plt

MAX_ITS = 70
MAX_TOL = 1.0e-8
REL_TOL = 1.0e-3

def cg_simple(A, b, x):

    file = open("cg_simple.dat", "w") 

    x.fill(0.0)
    r = b - np.dot(A, x)
    norm = np.linalg.norm(r)
    norm_0 = norm
    p = r
    its = 0

    while (its < MAX_ITS):

        print "it = ", its, "norm = ", norm
        file.write(str(its) + " " + str(norm) + "\n") 

        rr = np.dot(r, r)
        Ap = np.dot(A, p)
        alpha = rr / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        norm = np.linalg.norm(r)

        if (norm < MAX_TOL or norm < norm_0 * REL_TOL):
            break

        beta = np.dot(r, r) / rr
        p = r + beta * p
        its += 1

    file.close()
    return

#------------------------------------------------------------

def cg_pd(A, b, x):

    file = open("cg_pd.dat", "w") 

    x.fill(0.0)
    r = b - np.dot(A, x)
    norm = np.linalg.norm(r)
    norm_0 = norm

    M_inv = np.reciprocal(np.diagonal(A))
    z = np.multiply(M_inv, r)
    p = z
    its = 0

    while (its < MAX_ITS):

        print "it = ", its, "norm = ", norm
        file.write(str(its) + " " + str(norm) + "\n") 

        rz = np.dot(r, z)
        Ap = np.dot(A, p)
        alpha = rz / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        norm = np.linalg.norm(r)

        if (norm < MAX_TOL or norm < norm_0 * REL_TOL):
            break

        z = np.multiply(M_inv, r)
        beta = np.dot(r, z) / rz
        p = z + beta * p

        its += 1

    file.close()
    return
