#ifndef SymmetricEigen_h_
#define SymmetricEigen_h_

/**
 * Copyright © 2018, Andy Somogyi, andy -dot- somogyi -at- gmail you know...
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * Collection of three different implementation of eigen solvers optimized for the
 * 3x3 symmetric matricies we often find in computer graphics and simulation, such
 * as intertia tensors.
 *
 * This file provides a common interface for the three different solvers.
 */

#include "Magnum/Math/Matrix.h"
#include <math.h>
#include "MxDebug.h"
#include "GteSymmetricEigensolver3x3.h"

namespace Magnum { namespace Math { namespace Algorithms {

namespace Implementation {

/* Implementation of Eigen-decomposition for symmetric 3x3 real matrices.
   Public domain, copied from the public domain Java library JAMA. */

template<class T1, class T2>
T1 hypot2(T1 x, T2 y) {
    return std::sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.
template<class T, int n = 3>
void tred2(Matrix3<T> &V, Vector3<T> &d, Vector3<T> &e) {

    //  This is derived from the Algol procedures tred2 by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    for (int j = 0; j < n; j++) {
        d[j] = V[n-1][j];
    }

    // Householder reduction to tridiagonal form.

    for (int i = n-1; i > 0; i--) {

        // Scale to avoid under/overflow.

        T scale = 0.0;
        T h = 0.0;
        for (int k = 0; k < i; k++) {
            scale = scale + fabs(d[k]);
        }
        if (scale == 0.0) {
            e[i] = d[i-1];
            for (int j = 0; j < i; j++) {
                d[j] = V[i-1][j];
                V[i][j] = 0.0;
                V[j][i] = 0.0;
            }
        } else {

            // Generate Householder vector.

            for (int k = 0; k < i; k++) {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            T f = d[i-1];
            T g = sqrt(h);
            if (f > 0) {
                g = -g;
            }
            e[i] = scale * g;
            h = h - f * g;
            d[i-1] = f - g;
            for (int j = 0; j < i; j++) {
                e[j] = 0.0;
            }

            // Apply similarity transformation to remaining columns.

            for (int j = 0; j < i; j++) {
                f = d[j];
                V[j][i] = f;
                g = e[j] + V[j][j] * f;
                for (int k = j+1; k <= i-1; k++) {
                    g += V[k][j] * d[k];
                    e[k] += V[k][j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for (int j = 0; j < i; j++) {
                e[j] /= h;
                f += e[j] * d[j];
            }
            T hh = f / (h + h);
            for (int j = 0; j < i; j++) {
                e[j] -= hh * d[j];
            }
            for (int j = 0; j < i; j++) {
                f = d[j];
                g = e[j];
                for (int k = j; k <= i-1; k++) {
                    V[k][j] -= (f * e[k] + g * d[k]);
                }
                d[j] = V[i-1][j];
                V[i][j] = 0.0;
            }
        }
        d[i] = h;
    }

    // Accumulate transformations.

    for (int i = 0; i < n-1; i++) {
        V[n-1][i] = V[i][i];
        V[i][i] = 1.0;
        T h = d[i+1];
        if (h != 0.0) {
            for (int k = 0; k <= i; k++) {
                d[k] = V[k][i+1] / h;
            }
            for (int j = 0; j <= i; j++) {
                T g = 0.0;
                for (int k = 0; k <= i; k++) {
                    g += V[k][i+1] * V[k][j];
                }
                for (int k = 0; k <= i; k++) {
                    V[k][j] -= g * d[k];
                }
            }
        }
        for (int k = 0; k <= i; k++) {
            V[k][i+1] = 0.0;
        }
    }
    for (int j = 0; j < n; j++) {
        d[j] = V[n-1][j];
        V[n-1][j] = 0.0;
    }
    V[n-1][n-1] = 1.0;
    e[0] = 0.0;
}

// Symmetric tridiagonal QL algorithm.

template<class T, int n = 3>
void tql2(Matrix3<T> &V, Vector3<T> &d, Vector3<T> &e) {

    //  This is derived from the Algol procedures tql2, by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    for (int i = 1; i < n; i++) {
        e[i-1] = e[i];
    }
    e[n-1] = 0.0;

    T f = 0.0;
    T tst1 = 0.0;
    T eps = pow(2.0,-52.0);
    for (int l = 0; l < n; l++) {

        // Find small subdiagonal element

        tst1 = std::max(tst1,fabs(d[l]) + fabs(e[l]));
        int m = l;
        while (m < n) {
            if (fabs(e[m]) <= eps*tst1) {
                break;
            }
            m++;
        }

        // If m == l, d[l] is an eigenvalue,
        // otherwise, iterate.

        if (m > l) {
            int iter = 0;
            do {
                iter = iter + 1;  // (Could check iteration count here.)

                // Compute implicit shift

                T g = d[l];
                T p = (d[l+1] - g) / (2.0 * e[l]);
                T r = hypot2(p,1.0);
                if (p < 0) {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l+1] = e[l] * (p + r);
                T dl1 = d[l+1];
                T h = g - d[l];
                for (int i = l+2; i < n; i++) {
                    d[i] -= h;
                }
                f = f + h;

                // Implicit QL transformation.

                p = d[m];
                T c = 1.0;
                T c2 = c;
                T c3 = c;
                T el1 = e[l+1];
                T s = 0.0;
                T s2 = 0.0;
                for (int i = m-1; i >= l; i--) {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot2(p,e[i]);
                    e[i+1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i+1] = h + s * (c * g + s * d[i]);

                    // Accumulate transformation.

                    for (int k = 0; k < n; k++) {
                        h = V[k][i+1];
                        V[k][i+1] = s * V[k][i] + c * h;
                        V[k][i] = c * V[k][i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                // Check for convergence.

            } while (fabs(e[l]) > eps*tst1);
        }
        d[l] = d[l] + f;
        e[l] = 0.0;
    }

    // Sort eigenvalues and corresponding vectors.

    for (int i = 0; i < n-1; i++) {
        int k = i;
        T p = d[i];
        for (int j = i+1; j < n; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (int j = 0; j < n; j++) {
                p = V[j][i];
                V[j][i] = V[j][k];
                V[j][k] = p;
            }
        }
    }
}

}


/**
 *  Symmetric matrix A => eigenvectors in columns of V, corresponding
 *  eigenvalues in d.
 */
template<class T, int n=3> void eigen3(const Matrix3<T>& A, Matrix3<T>& V, Vector3<T> &d) {
    Vector3<T> e;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i][j] = A[i][j];
        }
    }
    Implementation::tred2(V, d, e);
    Implementation::tql2(V, d, e);
}

template <class T, int n=3> std::pair<Matrix<n, T>, Vector3<T>>
        symmetricEigen3x3Iterative(T a00, T a01, T a02, T a11, T a12, T a22,
        bool aggressive=false, int sortType=0) {
    std::array<T, n> eval;
    std::array<std::array<T, n>, n> evec;

    gte::SymmetricEigensolver3x3<T> eisolv;

    eisolv(a00, a01, a02, a11, a12, a22, aggressive, sortType, eval, evec);

    Matrix3<T> V{{evec[0][0], evec[0][1], evec[0][2]},
                 {evec[1][0], evec[1][1], evec[1][2]},
                 {evec[2][0], evec[2][1], evec[2][2]}};

    Vector3<T> d{eval[0], eval[1], eval[2]};

    return std::make_pair(V, d);
}

template <class T, int n=3> std::pair<Matrix<n, T>, Vector3<T>>
        symmetricEigen3x3Iterative(const Matrix3<T>& A, bool aggressive=false, int sortType=0) {
    std::array<T, n> eval;
    std::array<std::array<T, n>, n> evec;

    gte::SymmetricEigensolver3x3<T> eisolv;

    eisolv(A[0][0], A[0][1], A[0][2], A[1][1], A[1][2], A[2][2], aggressive, sortType, eval, evec);

    Matrix3<T> V{{evec[0][0], evec[0][1], evec[0][2]},
                 {evec[1][0], evec[1][1], evec[1][2]},
                 {evec[2][0], evec[2][1], evec[2][2]}};

    Vector3<T> d{eval[0], eval[1], eval[2]};

    return std::make_pair(V, d);
}

template <class T, int n=3> std::pair<Matrix<n, T>, Vector3<T>>
        symmetricEigen3x3Analytic(T a00, T a01, T a02, T a11, T a12, T a22) {
    std::array<T, n> eval;
    std::array<std::array<T, n>, n> evec;

    gte::NISymmetricEigensolver3x3<T> eisolv;

    eisolv(a00, a01, a02, a11, a12, a22, eval, evec);

    Matrix3<T> V{{evec[0][0], evec[0][1], evec[0][2]},
                 {evec[1][0], evec[1][1], evec[1][2]},
                 {evec[2][0], evec[2][1], evec[2][2]}};

    Vector3<T> d{eval[0], eval[1], eval[2]};

    return std::make_pair(V, d);
}

template <class T, int n=3> std::pair<Matrix<n, T>, Vector3<T>>
        symmetricEigen3x3Analytic(const Matrix3<T>& A) {
    std::array<T, n> eval;
    std::array<std::array<T, n>, n> evec;

    gte::NISymmetricEigensolver3x3<T> eisolv;

    eisolv(A[0][0], A[0][1], A[0][2], A[1][1], A[1][2], A[2][2], eval, evec);

    Matrix3<T> V{{evec[0][0], evec[0][1], evec[0][2]},
                 {evec[1][0], evec[1][1], evec[1][2]},
                 {evec[2][0], evec[2][1], evec[2][2]}};

    Vector3<T> d{eval[0], eval[1], eval[2]};

    return std::make_pair(V, d);
}



template <class T, int n=3> std::pair<Matrix<n, T>, Vector3<T>>
        symmetricEigen3x3TriDiagonal(const Matrix3<T>& A) {
    Matrix3<T> V;
    Vector3<T> d;
    Vector3<T> e;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i][j] = A[i][j];
        }
    }
    Implementation::tred2(V, d, e);
    Implementation::tql2(V, d, e);

    return std::make_pair(V.transposed(), d);
}

template <class T, int n=3> std::pair<Matrix<n, T>, Vector<n, T>>
        symmetricEigen3x3TriDiagonal(T a00, T a01, T a02, T a11, T a12, T a22) {
    // construct a matrix from column vectors
    Matrix3<T> A = {{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}};
    return symmetricEigen3x3TriDiagonal(A);
}


}}}


#endif
