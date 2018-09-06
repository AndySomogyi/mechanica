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
 * @file
 *
 * Collection of three different implementation of eigen solvers optimized for the
 * 3x3 symmetric matrices we often find in computer graphics and simulation, such
 * as inertia tensors.
 *
 * In general, the eigen vectors of a square matrix define an orthogonal basis set.
 * In computer graphics and simulation, a common task is to find a local coordinate
 * for an object composed of discrete components where the inertia tensor is diagonal.
 * Say we have an object composed of particles (in world space), and we want to find
 * a natural local coordinate system. Here, we would calculate the inertia tensor
 * from each particle, then we would calculate the eigen vectors. These vectors are
 * by definition orthogonal, and define a set of basis vectors.
 *
 * For a 3x3 matrix, each of these functions returns a matrix of eigen vectors, where the
 * eigen vectors form the columns of the matrix.
 *
 * This matrix can be used a rotation matrix, to convert from world space to local
 * space. Furthermore, the inverse of a rotation matrix is it's transpose, so to
 * move from local space to world space, we simply transpose the matrix.
 *
 * Note that eigen vectors are not unique, certain solvers may return the vectors
 * scaled by a different constant, and even when  normalized, they vectors may
 * point in opposite directions. What's important however is that the set of eigen
 * vectors is orthogonal.
 *
 * Most eigen solver algorithms are designed for large matrices. These algorithms
 * are inefficient in real-time simulation applications, where we normally just need
 * to calculate the eigen system for a 3x3 symmetric matrix. This file provides two
 * different kinds of solvers: analytic and iterative. The analytic solver,
 * @brief Function @ref Magnum::Math::Algorithms::symmetricEigen3x3Analytic()
 * uses Cardano's method to calculate the eigen system. This solver is usually faster,
 * however is often less accurate than iterative solvers.
 *
 * The two iterative solvers, @ref Magnum::Math::Algorithms::symmetricEigen3x3TriDiagonal()
 * and @ref Magnum::Math::Algorithms::symmetricEigen3x3Iterative() use an iterative approach
 * (with a fixed maximum number of steps) to compute a solution. Users should experiment
 * which iterative solver works better for them, however there should be negligible
 * performance difference. It is unclear as to which iterative solver should be preferred
 * over the other.
 */

#include "Magnum/Math/Matrix.h"
#include <math.h>
#include "MxDebug.h"
#include "GteSymmetricEigensolver3x3.h"

namespace Magnum { namespace Math { namespace Algorithms {

namespace Implementation {

/**
 * Implementation of Eigen-decomposition for symmetric 3x3 real matrices.
 * Public domain, copied from the public domain Java library JAMA.
 */

template<class T1, class T2>
T1 hypot2(T1 x, T2 y) {
    return std::sqrt(x*x+y*y);
}

/**
 * Symmetric Householder reduction to tridiagonal form.
 */
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


/**
 * Symmetric tridiagonal QL algorithm.
 */
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
 * @brief Calculates the eigenvectors and eigenvalues of a symmetric 3x3
 * matrix using an iterative algorithm.
 *
 * Because the input matrix must be symmetric, the unique elements are
 * a00, a01, a02, a11, a12, and a22. Note, this function does not verify
 * that the matrix is symmetric, however it only uses the a00, a01, a02,
 * a11, a12, and a22 elements from the input matrix, all other entries
 * are ignored.
 *
 * if 'aggressive' is 'true', the iterations occur until a superdiagonal
 * entry is exactly zero.  If 'aggressive' is 'false', the iterations
 * occur until a superdiagonal entry is effectively zero compared to the
 * sum of magnitudes of its diagonal neighbors.  Generally, the
 * nonaggressive convergence is acceptable.
 *
 * The order of the eigenvalues is specified by sortType: -1 (decreasing),
 * 0 (no sorting), or +1 (increasing).  When sorted, the eigenvectors are
 * ordered accordingly, and {evec[0], evec[1], evec[2]} is guaranteed to
 * be a right-handed orthonormal set.  The return value is the number of
 * iterations used by the algorithm.
 *
 * This function wraps David Eberly, Geometric Tools, iterative eigen solver.
 * @see http://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
 * describes algorithms for solving the eigensystem associated with a 3x3
 * symmetric real-valued matrix.
 *
 * It is a based on modified Symmetric QR Algorithm, and uses Householder
 * Tridiagonalization to reduce matrix A to tridiagonal form. Then uses
 * Implicit Symmetric QR Step with Wilkinson Shift for the iterative reduction
 * from tridiagonal to diagonal.
 */
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


/**
 * @brief Calculates the eigenvectors and eigenvalues of a symmetric 3x3
 * matrix using an iterative algorithm.
 *
 * Because the input matrix must be symmetric, the unique elements are
 * a00, a01, a02, a11, a12, and a22. Note, this function does not verify
 * that the matrix is symmetric, however it only uses the a00, a01, a02,
 * a11, a12, and a22 elements from the input matrix, all other entries
 * are ignored.
 *
 * if 'aggressive' is 'true', the iterations occur until a superdiagonal
 * entry is exactly zero.  If 'aggressive' is 'false', the iterations
 * occur until a superdiagonal entry is effectively zero compared to the
 * sum of magnitudes of its diagonal neighbors.  Generally, the
 * nonaggressive convergence is acceptable.
 *
 * The order of the eigenvalues is specified by sortType: -1 (decreasing),
 * 0 (no sorting), or +1 (increasing).  When sorted, the eigenvectors are
 * ordered accordingly, and {evec[0], evec[1], evec[2]} is guaranteed to
 * be a right-handed orthonormal set.  The return value is the number of
 * iterations used by the algorithm.
 *
 * This function wraps David Eberly, Geometric Tools, iterative eigen solver.
 * @see http://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
 * describes algorithms for solving the eigensystem associated with a 3x3
 * symmetric real-valued matrix.
 *
 * It is a based on modified Symmetric QR Algorithm, and uses Householder
 * Tridiagonalization to reduce matrix A to tridiagonal form. Then uses
 * Implicit Symmetric QR Step with Wilkinson Shift for the iterative reduction
 * from tridiagonal to diagonal.
 */
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

/**
 * @brief Calculates the eigenvectors and eigenvalues of a symmetric 3x3
 * matrix using an analytic algorithm.
 *
 * Because the input matrix must be symmetric, the unique elements are
 * a00, a01, a02, a11, a12, and a22. Note, this function does not verify
 * that the matrix is symmetric, however it only uses the a00, a01, a02,
 * a11, a12, and a22 elements from the input matrix, all other entries
 * are ignored.
 *
 * Calculates the analytic solution based on Cardanos's method.
 *
 * This function wraps David Eberly, Geometric Tools, analytic eigen solver.
 * @see http://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
 * describes algorithms for solving the eigensystem associated with a 3x3
 * symmetric real-valued matrix.
 */
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

/**
 * @brief Calculates the eigenvectors and eigenvalues of a symmetric 3x3
 * matrix using an analytic algorithm.
 *
 * Because the input matrix must be symmetric, the unique elements are
 * a00, a01, a02, a11, a12, and a22.
 *
 * Calculates the analytic solution based on Cardanos's method.
 *
 * This function wraps David Eberly, Geometric Tools, analytic eigen solver.
 * @see http://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
 * describes algorithms for solving the eigensystem associated with a 3x3
 * symmetric real-valued matrix.
 */
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



/**
 * @brief Calculates the eigenvectors and eigenvalues of a symmetric 3x3 matrix.
 *
 * Because the input matrix must be symmetric, the unique elements are
 * a00, a01, a02, a11, a12, and a22.
 *
 * Based on code from  Giorgio Grisetti, Cyrill Stachniss, Wolfram Burgard,
 * @see http://docs.ros.org/kinetic/api/openslam_gmapping/html/eig3_8cpp_source.html
 *
 * This is a similar approach to David Eberly, but does not return normalized eigen
 * vectors.
 *
 * A implementation of a tri-diagonal eigen solver code for symmetric 3x3 matrices,
 * copied from the public domain Java Matrix library JAMA. This in turn was
 * This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch,
 * and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra, and
 * the corresponding Fortran subroutine in EISPACK.
 */
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


/**
 * @brief Calculates the eigenvectors and eigenvalues of a symmetric 3x3 matrix.
 *
 * Because the input matrix must be symmetric, the unique elements are
 * a00, a01, a02, a11, a12, and a22. Note, this function does not verify
 * that the matrix is symmetric, however it only uses the a00, a01, a02,
 * a11, a12, and a22 elements from the input matrix, all other entries
 * are ignored.
 *
 * Based on code from  Giorgio Grisetti, Cyrill Stachniss, Wolfram Burgard,
 * @see http://docs.ros.org/kinetic/api/openslam_gmapping/html/eig3_8cpp_source.html
 *
 * This is a similar approach to David Eberly, but does not return normalized eigen
 * vectors.
 *
 * A implementation of a tri-diagonal eigen solver code for symmetric 3x3 matrices,
 * copied from the public domain Java Matrix library JAMA. This in turn was
 * This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch,
 * and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra, and
 * the corresponding Fortran subroutine in EISPACK.
 */
template <class T, int n=3> std::pair<Matrix<n, T>, Vector<n, T>>
        symmetricEigen3x3TriDiagonal(T a00, T a01, T a02, T a11, T a12, T a22) {
    // construct a matrix from column vectors
    Matrix3<T> A = {{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}};
    return symmetricEigen3x3TriDiagonal(A);
}


}}}


#endif
