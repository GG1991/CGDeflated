#
# Conjugate Gradient Algorithms
#
#
#

import numpy as np
import matplotlib.pyplot as plt

MAX_ITS = 4
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

#------------------------------------------------------------

def cg_deflated(A, b, x, ngroups=2):

    file = open("cg_deflated.dat", "w")
    print "ngroups = ", ngroups

    n = x.shape[0]
    A_coarse = np.zeros((ngroups, ngroups))
    lgroup = np.zeros(n)
    lgroup = lgroup.astype(int)

    for i in range(0, n):
        lgroup[i] = i / (n / ngroups)
    lgroup[0] = -1
    lgroup[n - 1] = -1

    print "n = ", n, "n / ngroup = ", (n / ngroups)
    print lgroup

    # Construct coarse matrix
    for i in range(0, n):
        ig = lgroup[i]
        if (ig >= 0):
            for j in range(0, n):
                jg = lgroup[j]
                if (jg >= 0):
                    A_coarse[ig, jg] += A[i, j]

    x.fill(0.0)
    r = b - np.dot(A, x)

    # A_coarse x d = W^T r = r_coarse
    r_coarse = np.zeros(ngroups)
    for i in range(0, ngroups):
        r_coarse[lgroup[i]] += r[i]

    print A_coarse
    d_coarse = np.linalg.solve(A_coarse, r_coarse)
    #print "d_coarse", d_coarse

    d_fine = np.zeros(n)
    for i in range(0, n):
        ig = lgroup[i]
        if (ig >= 0):
            d_fine[i] += d_coarse[ig]

    x += d_fine
    #print "x", x

    r = b - np.dot(A, x)

    norm = np.linalg.norm(r)
    norm_0 = norm

    M_inv = np.reciprocal(np.diagonal(A))
    z = np.multiply(M_inv, r)

    # A_coarse x d = W^T A z = r_coarse
    Az = np.dot(A,z)
    r_coarse.fill(0)
    for i in range(0, ngroups):
        r_coarse[lgroup[i]] += Az[i]
    d_coarse = np.linalg.solve(A_coarse, r_coarse)

    d_fine.fill(0)
    for i in range(0, n):
        ig = lgroup[i]
        if (ig >= 0):
            d_fine[i] += d_coarse[ig]

    p = -d_fine + z
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

        # A_coarse x d_coarse = W^T A z = r_coarse
        Az = np.dot(A,z)
        r_coarse.fill(0)
        for i in range(0, ngroups):
            r_coarse[lgroup[i]] += Az[i]
        d_coarse = np.linalg.solve(A_coarse, r_coarse)

        d_fine.fill(0)
        for i in range(0, n):
            ig = lgroup[i]
            if (ig >= 0):
                d_fine[i] += d_coarse[ig]

        p = z + beta * p - d_fine

        its += 1

    file.close()
    return
