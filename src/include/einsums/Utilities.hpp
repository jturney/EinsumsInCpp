#pragma once

#include "Tensor.hpp"

namespace einsums {

template <typename T>
auto arange(T start, T stop, T step = T{1}) -> Tensor<T, 1> {
    assert(stop >= start);

    // Determine the number of elements that will be produced.
    int nelem = static_cast<int>((stop - start) / step);

    auto result = create_tensor<T>("arange created tensor", nelem);
    zero(result);

    int index = 0;
    for (T value = start; value < stop; value += step) {
        result(index++) = value;
    }

    return result;
}

template <typename T>
auto arange(T stop) -> Tensor<T, 1> {
    return arange(T{0}, stop);
}

} // namespace einsums