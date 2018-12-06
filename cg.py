#
# Conjugate Gradient Algorithms
#
#
#

import numpy as np
import matplotlib.pyplot as plt

MAX_ITS = 200
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

def fine_to_coarse(fine, coarse, lgroup):

    n = fine.shape[0]
    coarse.fill(0)
    for i in range(0, n):
        ig = lgroup[i]
        if (ig >= 0):
            coarse[ig] += fine[i]
    return

def coarse_to_fine(coarse, fine, lgroup):

    n = fine.shape[0]
    fine.fill(0)
    for i in range(0, n):
        ig = lgroup[i]
        if (ig >= 0):
            fine[i] += coarse[ig]
    return

#------------------------------------------------------------

def cg_deflated(A, b, x, ngroups=2):

    file = open("cg_deflated.dat", "w")

    n = x.shape[0]
    A_coarse = np.zeros((ngroups, ngroups))
    lgroup = np.zeros(n)
    lgroup = lgroup.astype(int)

    r_coarse = np.empty(ngroups)
    d_coarse = np.empty(ngroups)
    d_fine = np.empty(n)

    for i in range(0, n):
        lgroup[i] = i / (n / ngroups + 1)
    lgroup[0] = -1
    lgroup[n - 1] = -1

    print "n = ", n, "\telements per group = ", (n / ngroups)

    # Construct coarse matrix
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            A_coarse[lgroup[i], lgroup[j]] += A[i, j]

    x.fill(0)

    # A_coarse x d = W^T r = r_coarse
    r = b - np.dot(A, x)
    fine_to_coarse(r, r_coarse, lgroup)
    d_coarse = np.linalg.solve(A_coarse, r_coarse)
    coarse_to_fine(d_coarse, d_fine, lgroup)

    x += d_fine
    r = b - np.dot(A, x)

    norm = np.linalg.norm(r)
    norm_0 = norm

    M_inv = np.reciprocal(np.diagonal(A))
    z = np.multiply(M_inv, r)

    # A_coarse x d = W^T A z = r_coarse
    Az = np.dot(A,z)
    fine_to_coarse(Az, r_coarse, lgroup)
    d_coarse = np.linalg.solve(A_coarse, r_coarse)
    coarse_to_fine(d_coarse, d_fine, lgroup)

    p = z - d_fine
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
        fine_to_coarse(Az, r_coarse, lgroup)
        d_coarse = np.linalg.solve(A_coarse, r_coarse)
        coarse_to_fine(d_coarse, d_fine, lgroup)

        p = z + beta * p - d_fine

        its += 1

    file.close()
    return
