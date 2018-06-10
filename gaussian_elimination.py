# ---------------------------------------------
#           Gaussian elemination
#        with complete pivot selection 
# ---------------------------------------------
import numpy as np
from numpy import sqrt, real, imag, zeros, array, abs, sum, float64, float128


# ---------------------------------------------
def complete_pivot_selection(n: int, a: float64, b: float64, ib: int, k: int):
    """
    int     n       :  Matrix size
    double a[n,n], b[n], x[n]
    ##
    double  abskk   :Absolute value of pivot
    int     ik, jk  :Position of pivot
    int     ibw     :Temp. for ib[]
    double  aw, bw  :Temp. for a, b
    int     i, j
    double  akk
    """
    # print('a=', a)
    # print('b=', b)

    # Initialize index
    if k == 0:
        for i in range(n):
            ib[i] = i

    # Select pivot
    ik = k
    jk = k
    abskk = abs(a[k, k])
    for j in range(k, n):
        for i in range(k, n):
            akk = a[k, k]
            if abs(akk) > abskk:
                abskk = abs(akk)
                ik = i
                jk = j

    # Swap of rows
    if ik != k:
        for j in range(k, n):  # a
            aw = a[k, j]
            a[k, j] = a[ik, j]
            a[ik, j] = aw
        bw = b[k]  # b
        b[k] = b[ik]
        b[ik] = bw

    # Swap of columns
    if jk != k:
        for i in range(n):  # a
            aw = a[i, k]
            a[i, k] = a[i, jk]
            a[i, jk] = aw
        ibw = ib[k]  # b
        ib[k] = ib[jk]
        ib[jk] = ibw

    # Return pivot
    eps = 1e-14
    if abs(akk) < eps:
        print('k=', k)
        print("Matrix singular, Pivot =( %1.000e" % real(akk), ",", "%1.000e)\n" % imag(akk))
        err = 1
        return a[k, k]
    return a[k, k]


# --------------------------------------------
def gaussian_elimination(n: int, a: float64, b: float64, x: float64):
    """
    Try to solve linear equation with float64 precision:   a * x = b
    int n        : Matrix size
    float        : a[n,n], b[n], x[n]
    return：err  : 0=Normal end, 1=Matrix singular
    #
    int     ib[N]      # index for lineup
    int     i, j, k
    double  akk
    double  aik
    """

    # Forward erasing
    eps = 1.e-14  # Matrix singular
    ib = zeros(n, dtype=int)

    for k in range(n):
        # Select complete pivot akk
        akk = complete_pivot_selection(n, a, b, ib, k)

        # Normalize row with the pivot akk
        for j in range(k, n):
            a[k, j] /= akk
        b[k] /= akk

        # Forward elimination to upper triangular matrix
        for i in range(k + 1, n):
            aik = a[i, k]
            for j in range(k, n):
                a[i, j] -= aik * a[k, j]
            b[i] -= aik * b[k]

    # Backsubstitution
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            b[i] -= a[i, j] * b[j]

    # Lineup b with ib
    for k in range(n):
        x[ib[k]] = b[k]

    print('a.dtype=', a.dtype)
    # print('b.dtype=', b.dtype)
    # print('x.dtype=', x.dtype)
    # print('ib.dtype=', ib.dtype)

    err = 0
    return err


# ---------------------------------------------
def make_random_matrix(n: int, a: float64, b: float64, x0: float64):
    """
     Make random matrix for test.
     x[n,1]= a[n,n] * b[n,1]
    """
    t = np.random.randn(n, n)
    a[:, :] = t.astype(np.float64)

    for i in range(n):
        x0[i] = (i + 1.0)

    for i in range(n):
        b[i] = 0.0
        for j in range(n):
            b[i] = b[i] + a[i, j] * x0[j]

        # print('make_random_matrix:a=', a)
        # print('make_random_matrix:b=', b)
        # print('make_random_matrix:x0=', x0)
        # print('make_random_matrix:a.dtype=', a.dtype)
        # print('make_random_matrix:b.dtype=', b.dtype)
        # print('make_random_matrix:x0.dtype=', x0.dtype)

# ---------------------------------------------
def make_hilbert_matrix(n: int, a: float64, b: float64, x0: float64):
    """
     Make ill conditioned matrix.
     b[n]= a[n,n] * x[n]
    """

    for i in range(n):
        for j in range(n):
            a[i, j] = 1 / (i + j + 1)

    for i in range(n):
        x0[i] = i + 1

    for i in range(n):
        b[i] = 0
        for j in range(n):
            b[i] = b[i] + a[i, j] * x0[j]

    # print('a.dtype=', a.dtype)
    # print('b.dtype=', b.dtype)
    # print('x0.dtype=', x0.dtype)


# ---------------------------------------------
def complete_pivot_selection_float128(n: int, a: float128, b: float128, ib: int, k: int):
    """
    int     n       :  Matrix size
    double a[n,n], b[n], x[n]
    ##
    double  abskk   :Absolute value of pivot
    int     ik, jk  :Position of pivot
    int     ibw     :Temp. for ib[]
    double  aw, bw  :Temp. for a, b
    int     i, j
    double  akk
    """
    # print('a=', a)
    # print('b=', b)

    # Initialize index
    if k == 0:
        for i in range(n):
            ib[i] = i

    # Select pivot
    ik = k
    jk = k
    abskk = abs(a[k, k])
    for j in range(k, n):
        for i in range(k, n):
            akk = a[k, k]
            if abs(akk) > abskk:
                abskk = abs(akk)
                ik = i
                jk = j

    # Swap of rows
    if ik != k:
        for j in range(k, n):  # a
            aw = a[k, j]
            a[k, j] = a[ik, j]
            a[ik, j] = aw
        bw = b[k]  # b
        b[k] = b[ik]
        b[ik] = bw

    # Swap of columns
    if jk != k:
        for i in range(n):  # a
            aw = a[i, k]
            a[i, k] = a[i, jk]
            a[i, jk] = aw
        ibw = ib[k]  # b
        ib[k] = ib[jk]
        ib[jk] = ibw

    # Return pivot
    eps = 1e-32
    if abs(akk) < eps:
        print('k=', k)
        print("Matrix singular, Pivot =( %1.000e" % real(akk), ",", "%1.000e)\n" % imag(akk))
        err = 1
        return a[k, k]
    return a[k, k]


# --------------------------------------------
def gaussian_elimination_float128(n: int, a: float128, b: float128, x: float128):
    """
    Try to solve linear equation with float128 precision:   a * x = b
    int n        : Matrix size
    float        : a[n,n], b[n], x[n]
    return：err  : 0=Normal end, 1=Matrix singular
    #
    int     ib[N]      # index for lineup
    int     i, j, k
    double  akk
    double  aik
    """

    # Forward erasing
    eps = 1.e-32  # Matrix singular
    ib = zeros(n, dtype=int)

    for k in range(n):
        # Select complete pivot akk
        akk = complete_pivot_selection(n, a, b, ib, k)

        # Normalize row with the pivot akk
        for j in range(k, n):
            a[k, j] /= akk
        b[k] /= akk

        # Forward elimination to upper triangular matrix
        for i in range(k + 1, n):
            aik = a[i, k]
            for j in range(k, n):
                a[i, j] -= aik * a[k, j]
            b[i] -= aik * b[k]

    # Backsubstitution
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            b[i] -= a[i, j] * b[j]

    # Lineup b with ib
    for k in range(n):
        x[ib[k]] = b[k]

    print('a.dtype=', a.dtype)
    # print('b.dtype=', b.dtype)
    # print('x.dtype=', x.dtype)
    # print('ib.dtype=', ib.dtype)

    err = 0
    return err


# ---------------------------------------------
def make_random_matrix_float128(n: int, a: float128, b: float128, x0: float128):
    """
     Make random matrix for test.
     x[n,1]= a[n,n] * b[n,1]
    """
    t = np.random.randn(n, n)
    a[:, :] = t.astype(np.float128)

    for i in range(n):
        x0[i] = (i + 1.0)

    for i in range(n):
        b[i] = 0.0
        for j in range(n):
            b[i] = b[i] + a[i, j] * x0[j]

        # print('make_random_matrix:a=', a)
        # print('make_random_matrix:b=', b)
        # print('make_random_matrix:x0=', x0)
        # print('make_random_matrix:a.dtype=', a.dtype)
        # print('make_random_matrix:b.dtype=', b.dtype)
        # print('make_random_matrix:x0.dtype=', x0.dtype)

# ---------------------------------------------
def make_hilbert_matrix_float128(n: int, a: float128, b: float128, x0: float128):
    """
     Make ill conditioned matrix.
     b[n]= a[n,n] * x[n]
    """

    for i in range(n):
        for j in range(n):
            a[i, j] = 1 / (i + j + 1)

    for i in range(n):
        x0[i] = i + 1

    for i in range(n):
        b[i] = 0
        for j in range(n):
            b[i] = b[i] + a[i, j] * x0[j]

    # print('a.dtype=', a.dtype)
    # print('b.dtype=', b.dtype)
    # print('x0.dtype=', x0.dtype)


# ---------------------------------------------
def main():
    """
    Solve the linear equation 'a x = b' for x.
    where   a[n,n], x[n], b[n]
    """

    n = 10  # Matrix size n x n
    # system = ('Windows')
    system = ('Mac_float64')
    system = ('Mac_float128')
    if ('Mac') in system:
        a = array(zeros((n, n), dtype='float64'))
        b = array(zeros(n, dtype='float64'))
        x = array(zeros(n, dtype='float64'))
        x0 = array(zeros(n, dtype='float64'))
        make_random_matrix(n, a, b, x0)
        # make_hilbert_matrix(n, a, b, x0)
        print('Mac:gaussian_elimination_float64')
        gaussian_elimination(n, a, b, x)
        er_norm = sqrt(sum(abs(x - x0) ** 2))
        print('er_norm_float64=%5.2e' % er_norm)

    if ('Mac') in system:
        a = array(zeros((n, n), dtype='float128'))
        b = array(zeros(n, dtype='float128'))
        x = array(zeros(n, dtype='float128'))
        x0 = array(zeros(n, dtype='float128'))
        make_random_matrix_float128(n, a, b, x0)
        # make_hilbert_matrix_float128(n, a, b, x0)
        print('Mac:gaussian_elimination_float128')
        gaussian_elimination_float128(n, a, b, x)
        er_norm = sqrt(sum(abs(x - x0) ** 2))
        print('er_norm_float128=%5.2e' % er_norm)

    if ('Windows') in system:
        # The python of Windows10 does't support float128.
        e = zeros((n, n), dtype='float64')
        f = zeros(n, dtype='float64')
        y = zeros(n, dtype='float64')
        y0 = zeros(n, dtype='float64')
        e = array(e)
        f = array(f)
        y = array(y)
        y0 = array(y0)
        # make_hilbert_matrix(n, a, b, x0)
        make_random_matrix(n, e, f, y0)
        y = np.linalg.solve(e, f)
        # The np.linalg of python does't support float128.
        er_norm = sqrt(sum(abs(y - y0) ** 2))
        print('Windows:np.linalg.solve')
        # print('x=', y)
        print('er_norm=%5.2e' % er_norm)
    return 0

# ---------------------------------------------
if __name__ == "__main__":
    code = main()
    print('\nexit code of main()=', code)
