#pragma once

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"

#include <stdexcept>

namespace einsums::polynomial {

template <typename T>
auto laguerre_companion(const Tensor<T, 1> &c) -> Tensor<T, 2> {
    if (c.dim(0) < 2) {
        throw std::runtime_error("Series must have maximum degree of at least 1.");
    }

    if (c.dim(0) == 2) {
        auto result = create_tensor<T>("Laguerre companion matrix", 1, 1);
        result(0, 0) = T{1} + (c(0) / c(1));
        return result;
    }

    auto n = c.dim(0) - 1;
    auto mat = create_tensor<T>("Laguerre companion matrix", n, n);
    zero(mat);

    auto mat1 = TensorView{mat, Dim<1>{-1}};
    auto top = TensorView{mat1, Dim<1>{-1}, Offset<1>{1}, Stride<1>{n + 1}};
    auto mid = TensorView{mat1, Dim<1>{-1}, Offset<1>{0}, Stride<1>{n + 1}};
    auto bot = TensorView{mat1, Dim<1>{-1}, Offset<1>{n}, Stride<1>{n + 1}};

    auto U = arange<double>(1, n);
    U *= -1.0;
    top = U;

    auto V = arange<double>(n);
    V *= 2.0;
    V += 1.0;
    mid = V;

    bot = U;

    /// TODO: Need to add c's contribution to mat. In python:
    /// mat[:, -1] += (c[:-1]/c[-1])*n

    return mat;
}

template <typename T>
auto laguerre_derivative(const Tensor<T, 1> &c, int m = 1, T scl = T{1}, int axis = 0) -> Tensor<T, 1> {
}

template <typename T>
auto laguerre_value(const Tensor<T, 1> &x, const Tensor<T, 1> &c) -> Tensor<T, 1> {
    auto c0 = create_tensor_like("c0", x), c1 = create_tensor_like("c1", x);
    zero(c0);
    zero(c1);

    if (c.dim(0) == 1) {
        c0 = c(0);
        c1 = T{0};
    } else if (c.dim(0) == 2) {
        c0 = c(0);
        c1 = c(1);
    } else {
        size_t nd = c.dim(0);
        // println("nd {}", nd);

        c0 = c(-2);
        c1 = c(-1);

        auto tmp = create_tensor_like("tmp", x);
        auto tmp1 = create_tensor_like("tmp1", x);
        for (int i = 3; i < c.dim(0) + 1; i++) {
            tmp = c0;
            nd = nd - 1;

            c0 = c1;
            c0 *= (nd - 1);
            c0 /= nd;
            c0 *= -1.0;
            c0 += c(-i);

            tmp1 = (2 * nd - 1);
            tmp1 -= x;
            tmp1 *= c1;
            tmp1 /= nd;
            c1 = tmp;
            c1 += tmp1;

            // println(tmp);
            // println(c0);
        }
    }

    // println("---");

    // println(x);
    // println(c0);
    // println(c1);

    auto result = create_tensor_like("result", x);
    result = T{1};
    // println(result);
    result -= x;
    // println(result);
    result *= c1;
    // println(result);
    result += c0;

    // println("adding c0 to result");
    // println(c0);

    // println("final result");
    // println(result);

    return result;
}

template <typename T = double>
auto laggauss(unsigned int degree) -> void /*std::tuple<Tensor<T, 1>, Tensor<T, 1>>*/ {
    // First approximation of roots. We use the fact that the companion matrix is symmetric in this case in order to obtain better zeros.
    auto c = create_tensor<double>("c", degree + 1);
    zero(c);
    c(-1) = 1.0;
    auto m = laguerre_companion(c);
    auto x = create_tensor<double>("x", degree);
    zero(x);

    println(m);
    println(x);
    linear_algebra::syev(&m, &x);

    // Improve roots by one application of Newtown.
    auto dy = laguerre_value(x, c);
    auto df = laguerre_value(x, laguerre_derivative(c));

    // x -= (dy / df);
}

} // namespace einsums::polynomial