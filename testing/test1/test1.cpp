#include  <iostream>

#include "MxModel.h"

#include <tuple>

#include <Magnum/Magnum.h>
#include "SymmetricEigen.h"

using namespace Magnum::Math;
using namespace Magnum::Math::Algorithms;


template<typename T>
Magnum::Math::Matrix3<T> diagMat(Magnum::Math::Vector3<T> &d) {
    Magnum::Math::Matrix3<T> result{{d[0], 0, 0}, {0, d[1], 0}, {0, 0, d[2]}};
    return result;
}


enum EigenSolver {
    Iterative, Analytic, TriDiagonal
};

template <EigenSolver type, typename T>
bool testEigenSolver(T a00, T a01, T a02, T a11, T a12, T a22) {
    Magnum::Math::Matrix3<T> V;
    Magnum::Math::Vector3<T> d;
    Magnum::Math::Matrix3<T> A = {{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}};

    switch (type) {
    case Iterative:
            std::cout << "Iterative Solver" << std::endl;
        std::tie(V,d) = Algorithms::symmetricEigen3x3Iterative(A[0][0], A[0][1], A[0][2], A[1][1], A[1][2], A[2][2]);
            break;
    case Analytic:
            std::cout << "Analytic Solver" << std::endl;
        std::tie(V,d) = Algorithms::symmetricEigen3x3Analytic(A[0][0], A[0][1], A[0][2], A[1][1], A[1][2], A[2][2]);
            break;
    case TriDiagonal:
            std::cout << "TriDiagonal Solver" << std::endl;
        std::tie(V,d) = Algorithms::symmetricEigen3x3TriDiagonal(A[0][0], A[0][1], A[0][2], A[1][1], A[1][2], A[2][2]);
            break;
        default:
            assert(0);
    }

    std::cout << "A: " << std::endl;
    std::cout << A << std::endl;

    std::cout << "eigen vectors Q:" << std::endl;
    std::cout << V << std::endl;

    std::cout << "eigen values d:" << std::endl;
    std::cout << d << std::endl << std::endl;

    Magnum::Matrix3 t = V * diagMat(d) * V.transposed();
    std::cout << "Q * d * Q^-1:" << std::endl;
    std::cout << t << std::endl;
    
    t = V.transposed() * diagMat(d) * V;
    std::cout << "Q^-1 * d * Q:" << std::endl;
    std::cout << t << std::endl;

    return true;
}



int main( int argc, char *argv[] ) {
    std::cout << "foo" << std::endl;

    Magnum::Matrix3 A{{1,2,3},{3,2,1},{1,0,-1}};

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            std::cout << "m[" << i << "][" << j << "]: " << A[i][j] << std::endl;
        }
    }

    for(int i = 0; i < 3; ++i) {
        std::cout << "row(" << i << "): " << A.row(i) << std::endl;
    }

    std::cout << "Iterative Solver:" << std::endl;
    testEigenSolver<Iterative, float>(1, 2, 3, 4, 5, 6);
    
    std::cout << "Analytic Solver:" << std::endl;
    testEigenSolver<Analytic, float>(1, 2, 3, 4, 5, 6);
    
    std::cout << "TriDiagonal Solver:" << std::endl;
    testEigenSolver<TriDiagonal, float>(1, 2, 3, 4, 5, 6);


     return 0;
}
