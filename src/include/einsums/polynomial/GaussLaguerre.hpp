#pragma once

#include "einsums/Tensor.hpp"

#include <stdexcept>

namespace einsums::polynomial {

template <typename T>
auto laguerre_companion(const Tensor<T, 1> &c) -> Tensor<T, 2> {
    if (c.dim(0) < 2) {
        throw std::runtime_error("Series must have maximum degree of at least 1.");
    }

    if (c.dim(0) == 2) {
        auto result = create_tensor<T>("Laguerre companion matrix", 1, 1);
        result(0, 0) = T{1} + (c[0] / c[1]);
        return result;
    }

    auto n = c.dim(0) - 1;
    auto mat = create_tensor<T>("Laguerre companion matrix", n, n);
    zero(mat);

    auto mat1 = TensorView{mat, Dim<1>{-1}};
    auto top = TensorView{mat1, Dim<1>{-1}, Offset<1>{1}, Stride<1>{n + 1}};
}

template <typename T = double>
auto gausslag(unsigned int degree) -> std::tuple<Tensor<T, 1>, Tensor<T, 1>> {
}

} // namespace einsums::polynomial