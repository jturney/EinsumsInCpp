#pragma once

#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/State.hpp"
#include "einsums/_Common.hpp"

// Include headers from the ranges library that we need to handle cartesian_products
#include "range/v3/range_fwd.hpp"
#include "range/v3/view/cartesian_product.hpp"
#include "range/v3/view/iota.hpp"

#include <H5Lpublic.h>
#include <H5Ppublic.h>
#include <H5public.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums {

namespace detail {

template <typename T, size_t Rank>
struct TensorBase {
    [[nodiscard]] virtual auto dim(int d) const -> size_t = 0;
};

} // namespace detail

template <typename T, size_t Rank>
struct TensorView;

template <typename T, size_t ViewRank, size_t Rank>
struct DiskView;

template <typename T, size_t Rank>
struct DiskTensor;

namespace detail {

template <template <typename, size_t> typename TensorType, size_t Rank, std::size_t... I, typename T>
auto get_dim_ranges(const TensorType<T, Rank> &tensor, std::index_sequence<I...>) {
    return std::tuple{ranges::views::ints(0, (int)tensor.dim(I))...};
}

template <size_t N, typename Target, typename Source1, typename Source2>
void add_elements(Target &target, const Source1 &source1, const Source2 &source2) {
    if constexpr (N > 1) {
        add_elements<N - 1>(target, source1, source2);
    }
    target[N - 1] = source1[N - 1] + std::get<N - 1>(source2);
}

} // namespace detail

template <int N, template <typename, size_t> typename TensorType, size_t Rank, typename T>
auto get_dim_ranges(const TensorType<T, Rank> &tensor) {
    return detail::get_dim_ranges(tensor, std::make_index_sequence<N>{});
}

/**
 * @brief The multidimensional tensor object.
 *
 * @tparam T Underlying data type to use. Generally, float, double, std::complex<float>, std::complex<double.
 * @tparam Rank The order of the tensor. E.g. 0 is a scalar, 1 a vector, 2 a matrix, etc.
 */
template <typename T, size_t Rank>
struct Tensor final : public detail::TensorBase<T, Rank> {

    using vector = std::vector<T, AlignedAllocator<T, 64>>;

    Tensor() = default;
    Tensor(const Tensor &) = default;
    // Tensor(Tensor &&) noexcept = default;
    ~Tensor() = default;

    template <typename... Dims>
    explicit Tensor(std::string name, Dims... dims) : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        struct stride {
            size_t value{1};
            stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), stride());
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

        // Resize the data structure
        // TODO: Is this setting all the elements to zero?
        _data.resize(size);
    }

    // Once this is called "otherTensor" is no longer a valid tensor.
    template <size_t OtherRank, typename... Dims>
    explicit Tensor(Tensor<T, OtherRank> &&existingTensor, std::string name, Dims... dims)
        : _data(std::move(existingTensor._data)), _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        static_assert(Rank == sizeof...(dims), "Declared rank does not match provided dims");

        struct stride {
            size_t value{1};
            stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we
        // compute this dimensionality of this "0" index for them.
        int nfound{0};
        int location{-1};
        for (auto [i, dim] : enumerate(_dims)) {
            if (dim == -1) {
                nfound++;
                location = i;
            }
        }

        if (nfound > 1) {
            throw std::runtime_error("More than one -1 was provided.");
        }

        if (nfound == 1) {
            size_t size{1};
            for (auto [i, dim] : enumerate(_dims)) {
                if (i != location)
                    size *= dim;
            }
            if (size > existingTensor.size()) {
                throw std::runtime_error("Size of new tensor is larger than the parent tensor.");
            }
            _dims[location] = existingTensor.size() / size;
        }

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), stride());
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

        // Check size
        if (_data.size() != size) {
            throw std::runtime_error("Provided dims to not match size of parent tensor");
        }
    }

    Tensor(Dim<Rank> dims) : _dims{std::move(dims)} {
        struct stride {
            size_t value{1};
            stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), stride());
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

        // Resize the data structure
        // TODO: Is this setting all the elements to zero?
        _data.resize(size);
    }

    Tensor(const TensorView<T, Rank> &other) : _name{other._name}, _dims{other._dims} {
        struct stride {
            size_t value{1};
            stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), stride());
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

        // Resize the data structure
        _data.resize(size);

        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target_value = std::apply(*this, target_combination);
            T value = std::apply(other, target_combination);
            target_value = value;
        }
    }

    void zero() {
#pragma omp parallel
        {
            auto tid = omp_get_thread_num();
            auto chunksize = _data.size() / omp_get_num_threads();
            auto begin = _data.begin() + chunksize * tid;
            auto end = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;
            std::fill(begin, end, 0);
        }
    }

    void set_all(T value) {
#pragma omp parallel
        {
            auto tid = omp_get_thread_num();
            auto chunksize = _data.size() / omp_get_num_threads();
            auto begin = _data.begin() + chunksize * tid;
            auto end = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;
            std::fill(begin, end, value);
        }
    }

    auto data() -> T * {
        return _data.data();
    }

    auto data() const -> const T * {
        return _data.data();
    }

    template <typename... MultiIndex>
    auto data(MultiIndex... index)
        -> std::enable_if_t<count_of_type<All_t, MultiIndex...>() == 0 && count_of_type<Range, MultiIndex...>() == 0, T *> {
        assert(sizeof...(MultiIndex) <= _dims.size());

        auto index_list = {static_cast<size_t>(index)...};
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return &_data[ordinal];
    }

    auto operator*=(const T &b) -> Tensor<T, Rank> & {
#pragma omp parallel
        {
            auto tid = omp_get_thread_num();
            auto chunksize = _data.size() / omp_get_num_threads();
            auto begin = _data.begin() + chunksize * tid;
            auto end = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;
#pragma omp simd
            for (auto i = begin; i < end; i++) {
                (*i) *= b;
            }
        }
        return *this;
    }

    auto operator+=(const T &b) -> Tensor<T, Rank> & {
#pragma omp parallel
        {
            auto tid = omp_get_thread_num();
            auto chunksize = _data.size() / omp_get_num_threads();
            auto begin = _data.begin() + chunksize * tid;
            auto end = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;
#pragma omp simd
            for (auto i = begin; i < end; i++) {
                (*i) += b;
            }
        }
        return *this;
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const
        -> std::enable_if_t<count_of_type<All_t, MultiIndex...>() == 0 && count_of_type<Range, MultiIndex...>() == 0, const T &> {
        assert(sizeof...(MultiIndex) == _dims.size());
        auto index_list = {static_cast<size_t>(index)...};
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return _data[ordinal];
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index)
        -> std::enable_if_t<count_of_type<All_t, MultiIndex...>() == 0 && count_of_type<Range, MultiIndex...>() == 0, T &> {
        assert(sizeof...(MultiIndex) == _dims.size());
        auto index_list = {static_cast<size_t>(index)...};
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return _data[ordinal];
    }

    // WARNING: Chances are this function will not work if you mix All{}, Range{} and explicit indexes.
    template <typename... MultiIndex>
    auto operator()(MultiIndex... index)
        -> std::enable_if_t<count_of_type<All_t, MultiIndex...>() >= 1,
                            TensorView<T, count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>> {
        // Construct a TensorView using the indices provided as the starting point for the view.
        // e.g.:
        //    Tensor T{"Big Tensor", 7, 7, 7, 7};
        //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
        // println("Here");
        const auto &indices = std::forward_as_tuple(index...);

        Offset<Rank> offsets;
        Stride<count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> strides{};
        Dim<count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims{};

        int counter{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            // println("looking at {}", i);
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, std::int64_t>) {
                auto tmp = static_cast<std::int64_t>(std::get<i>(indices));
                if (tmp < 0)
                    tmp = _dims[i] + tmp;
                offsets[i] = tmp;
            } else if constexpr (std::is_same_v<All_t, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                strides[counter] = _strides[i];
                dims[counter] = _dims[i];
                counter++;

            } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                auto range = std::get<i>(indices);
                offsets[counter] = range[0];
                if (range[1] < 0) {
                    auto temp = _dims[i] + range[1];
                    range[1] = temp;
                }
                dims[counter] = range[1] - range[0];
                strides[counter] = _strides[i];
                counter++;
            }
        });

        return TensorView<T, count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>{*this, std::move(dims), offsets,
                                                                                                            strides};
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> std::enable_if_t<count_of_type<Range, MultiIndex...>() == Rank, TensorView<T, Rank>> {
        Dim<Rank> dims{};
        Offset<Rank> offset{};
        Stride<Rank> stride = _strides;

        auto ranges = get_array_from_tuple<std::array<Range, Rank>>(std::forward_as_tuple(index...));

        for (int r = 0; r < Rank; r++) {
            auto range = ranges[r];
            offset[r] = range[0];
            if (range[1] < 0) {
                auto temp = _dims[r] + range[1];
                range[1] = temp;
            }
            dims[r] = range[1] - range[0];
        }

        return TensorView<T, Rank>{*this, std::move(dims), std::move(offset), std::move(stride)};
    }

    template <typename TOther>
    auto operator=(const Tensor<TOther, Rank> &other) -> Tensor<T, Rank> & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0)
                realloc = true;
            else if (dim(i) != other.dim(i)) {
                std::string str = fmt::format("Tensor::operator= dimensions do not match (this){} (other){}", dim(i), other.dim(i));
                if constexpr (Rank != 1)
                    throw std::runtime_error(str);
                else
                    realloc = true;
            }
        }

        if (realloc) {
            struct stride {
                size_t value{1};
                stride() = default;
                auto operator()(size_t dim) -> size_t {
                    auto old_value = value;
                    value *= dim;
                    return old_value;
                }
            };

            _dims = other._dims;

            // Row-major order of dimensions
            std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), stride());
            size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

            // Resize the data structure
            _data.resize(size);
        }

        if constexpr (std::is_same_v<T, TOther>) {
            std::copy(other._data.begin(), other._data.end(), _data.begin());
        } else {
            auto target_dims = get_dim_ranges<Rank>(*this);
            for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
                T &target_value = std::apply(*this, target_combination);
                T value = std::apply(other, target_combination);
                target_value = value;
            }
        }

        return *this;
    }

    template <typename TOther>
    auto operator=(const TensorView<TOther, Rank> &other) -> Tensor<T, Rank> & {
        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target_value = std::apply(*this, target_combination);
            T value = std::apply(other, target_combination);
            target_value = value;
        }

        return *this;
    }

    [[nodiscard]] auto dim(int d) const -> size_t {
        // Add support for negative indices.
        if (d < 0)
            d += Rank;
        return _dims[d];
    }
    auto dims() const -> Dim<Rank> {
        return _dims;
    }

    auto vector_data() const -> const vector & {
        return _data;
    }
    auto vector_data() -> vector & {
        return _data;
    }

    [[nodiscard]] auto name() const -> const std::string & {
        return _name;
    }
    void set_name(const std::string &name) {
        _name = name;
    }

    [[nodiscard]] auto stride(int d) const noexcept -> size_t {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    auto strides() const noexcept -> const auto & {
        return _strides;
    }

    auto to_rank_1_view() const -> TensorView<T, 1> {
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

        return TensorView<T, 1>{*this, dim};
    }

    // Returns the linear size of the tensor
    [[nodiscard]] auto size() const {
        return _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
    }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool {
        return true;
    }

  private:
    std::string _name{"(Unnamed)"};
    Dim<Rank> _dims;
    Stride<Rank> _strides;
    vector _data;

    template <typename T_, size_t Rank_>
    friend struct TensorView;

    template <typename T_, size_t OtherRank>
    friend struct Tensor;
};

template <>
struct Tensor<double, 0> : public detail::TensorBase<double, 0> {

    Tensor() = default;
    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    ~Tensor() = default;

    explicit Tensor(std::string name) : _name{std::move(name)} {};

    Tensor(Dim<0>) {}

    auto data() -> double * { return &_data; }
    [[nodiscard]] auto data() const -> const double * { return &_data; }

    auto operator=(const Tensor<double, 0> &other) -> Tensor<double, 0> & {
        _data = other._data;
        return *this;
    }

    auto operator=(const double &other) -> Tensor<double, 0> & {
        _data = other;
        return *this;
    }

    auto operator+=(const double &other) -> Tensor<double, 0> & {
        _data += other;
        return *this;
    }

    auto operator*=(const double &other) -> Tensor<double, 0> & {
        _data *= other;
        return *this;
    }

    auto operator/=(const double &other) -> Tensor<double, 0> & {
        _data /= other;
        return *this;
    }

    operator double() const { return _data; }

    [[nodiscard]] auto name() const -> const std::string & { return _name; }
    void set_name(const std::string &name) { _name = name; }

    [[nodiscard]] auto dim(int) const -> size_t override { return 1; }

    [[nodiscard]] auto dims() const -> Dim<0> { return Dim<0>{}; }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }

  private:
    std::string _name{"(Unnamed)"};
    double _data{};
};

template <typename T, size_t Rank>
struct TensorView final : public detail::TensorBase<T, Rank> {

    TensorView() = delete;
    TensorView(const TensorView &) = default;
    ~TensorView() = default;

    // std::enable_if doesn't work with constructors.  So we explicitly create individual
    // constructors for the types of tensors we support (Tensor and TensorView).  The
    // call to common_initialization is able to perform an enable_if check.
    template <size_t OtherRank, typename... Args>
    explicit TensorView(const Tensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        // println(" here 1");
        common_initialization(const_cast<Tensor<T, OtherRank> &>(other), args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit TensorView(Tensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        // println(" here 2");
        common_initialization(other, args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit TensorView(TensorView<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        // println(" here 3");
        common_initialization(other, args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit TensorView(const TensorView<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        // println(" here 4");
        common_initialization(const_cast<TensorView<T, OtherRank> &>(other), args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit TensorView(std::string name, Tensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{std::move(name)}, _dims{dim} {
        // println(" here 5");
        common_initialization(other, args...);
    }

    // template <typename... Args>
    // explicit TensorView(const double *other, const Dim<Rank> &dim) : _name{"Raw Array"}, _dims{dim} {
    //     common_initialization(other);
    // }

    auto operator=(const T *other) -> TensorView & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        auto target_dims = get_dim_ranges<Rank>(*this);
        size_t item{0};

        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target = std::apply(*this, target_combination);
            target = other[item];
            item++;
        }

        return *this;
    }

    template <template <typename, size_t> typename AType>
    auto operator=(const AType<T, Rank> &other) ->
        typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, Rank>, Rank, T>, TensorView &> {
        if constexpr (std::is_same_v<AType<T, Rank>, TensorView<T, Rank>>) {
            if (this == &other)
                return *this;
        }

        auto target_dims = get_dim_ranges<Rank>(*this);
        auto view = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for
        for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {
            T &target = std::apply(*this, *target_combination);
            target = std::apply(other, *target_combination);
        }

        return *this;
    }

    template <template <typename, size_t> typename AType>
    auto operator=(const AType<T, Rank> &&other) ->
        typename std::enable_if_t<is_incore_rank_tensor_v<AType<T, Rank>, Rank, T>, TensorView &> {
        if constexpr (std::is_same_v<AType<T, Rank>, TensorView<T, Rank>>) {
            if (this == &other)
                return *this;
        }

        auto target_dims = get_dim_ranges<Rank>(*this);
        auto view = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for
        for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {
            T &target = std::apply(*this, *target_combination);
            target = std::apply(other, *target_combination);
        }

        return *this;
    }

    auto operator=(const T &fill_value) -> TensorView & {
        auto target_dims = get_dim_ranges<Rank>(*this);
        auto view = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for
        for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {
            T &target = std::apply(*this, *target_combination);
            target = fill_value;
        }

        return *this;
    }

    auto operator/=(const T &value) -> TensorView & {
        auto target_dims = get_dim_ranges<Rank>(*this);
        auto view = std::apply(ranges::views::cartesian_product, target_dims);

#pragma omp parallel for
        for (auto target_combination = view.begin(); target_combination != view.end(); target_combination++) {
            T &target = std::apply(*this, *target_combination);
            target /= value;
        }

        return *this;
    }

    auto data() -> T * {
        return &_data[0];
    }
    auto data() const -> const T * {
        return static_cast<const T *>(&_data[0]);
    }
    template <typename... MultiIndex>
    auto data(MultiIndex... index) const -> T * {
        assert(sizeof...(MultiIndex) <= _dims.size());

        auto index_list = {std::forward<MultiIndex>(index)...};
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return &_data[ordinal];
    }

    auto data_array(const std::array<size_t, Rank> &index_list) const -> T * {
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return &_data[ordinal];
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> const T & {
        assert(sizeof...(MultiIndex) == _dims.size());
        auto index_list = {std::forward<MultiIndex>(index)...};
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return _data[ordinal];
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) -> T & {
        assert(sizeof...(MultiIndex) == _dims.size());
        auto index_list = {std::forward<MultiIndex>(index)...};
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return _data[ordinal];
    }

    [[nodiscard]] auto dim(int d) const -> size_t {
        if (d < 0)
            d += Rank;
        return _dims[d];
    }
    auto dims() const -> Dim<Rank> {
        return _dims;
    }

    [[nodiscard]] auto name() const -> const std::string & {
        return _name;
    }
    void set_name(const std::string &name) {
        _name = name;
    }

    [[nodiscard]] auto stride(int d) const noexcept -> size_t {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    auto strides() const noexcept -> const auto & {
        return _strides;
    }

    auto to_rank_1_view() const -> TensorView<T, 1> {
        if constexpr (Rank == 1) {
            return *this;
        } else {
            if (_strides[Rank - 1] != 1) {
                throw std::runtime_error("Creating a Rank-1 TensorView for this Tensor(View) is not supported.");
            }
            size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
            Dim<1> dim{size};

#if defined(EINSUMS_SHOW_WARNING)
            println("Creating a Rank-1 TensorView of an existing TensorView may not work. Be careful!");
#endif

            return TensorView<T, 1>{*this, dim, Stride<1>{1}};
        }
    }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool {
        return _full_view_of_underlying;
    }

  private:
    auto common_initialization(const T *other) {
        _data = const_cast<T *>(other);

        struct stride {
            size_t value{1};
            stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), stride());

        // At this time we'll assume we have full view of the underlying tensor since we were only provided
        // pointer.
        _full_view_of_underlying = true;
    }

    template <template <typename, size_t> typename TensorType, size_t OtherRank, typename... Args>
    auto common_initialization(TensorType<T, OtherRank> &other, Args &&...args)
        -> std::enable_if_t<std::is_base_of_v<detail::TensorBase<T, OtherRank>, TensorType<T, OtherRank>>> {

        static_assert(Rank <= OtherRank, "A TensorView must be the same Rank or smaller that the Tensor being viewed.");

        Stride<Rank> default_strides{};
        Offset<OtherRank> default_offsets{};
        Stride<Rank> error_strides{};
        error_strides[0] = -1;

        // If the Ranks are the same then use "other"s stride information
        if constexpr (Rank == OtherRank) {
            default_strides = other._strides;
            // Else since we're different Ranks we cannot automatically determine our stride and the user MUST
            // provide the information
        } else {
            if (std::accumulate(_dims.begin(), _dims.end(), 1.0, std::multiplies<>()) ==
                std::accumulate(other._dims.begin(), other._dims.end(), 1.0, std::multiplies<>())) {
                struct stride {
                    size_t value{1};
                    stride() = default;
                    auto operator()(size_t dim) -> size_t {
                        auto old_value = value;
                        value *= dim;
                        return old_value;
                    }
                };

                // Row-major order of dimensions
                std::transform(_dims.rbegin(), _dims.rend(), default_strides.rbegin(), stride());
                size_t size = default_strides.size() == 0 ? 0 : default_strides[0] * _dims[0];
            } else {
                // Stride information cannot be automatically deduced.  It must be provided.
                default_strides = Arguments::get(error_strides, args...);
                if (default_strides[0] == static_cast<size_t>(-1)) {
                    throw std::runtime_error("Unable to automatically deduce stride information. Stride must be passed in.");
                }
            }
        }

        default_offsets.fill(0);

        // Use default_* unless the caller provides one to use.
        _strides = Arguments::get(default_strides, args...);
        const Offset<OtherRank> &offsets = Arguments::get(default_offsets, args...);

        // Determine the ordinal using the offsets provided (if any) and the strides of the parent
        size_t ordinal = std::inner_product(offsets.begin(), offsets.end(), other._strides.begin(), size_t{0});
        _data = &(other._data[ordinal]);
    }

    std::string _name{"(Unnamed View)"};
    Dim<Rank> _dims;
    Stride<Rank> _strides;
    // Offsets<Rank> _offsets;

    bool _full_view_of_underlying{false};

    T *_data;

    template <typename T_, size_t Rank_>
    friend struct Tensor;

    template <typename T_, size_t OtherRank_>
    friend struct TensorView;
};

} // namespace einsums

// Include HDF5 interface
#include "H5.hpp"

// Tensor IO interface
namespace einsums {

template <size_t Rank, typename T, class... Args>
void write(const h5::fd_t &fd, const Tensor<T, Rank> &ref, Args &&...args) {
    // Can these h5 parameters be moved into the Tensor class?
    h5::current_dims current_dims_default;
    h5::max_dims max_dims_default;
    h5::count count_default;
    h5::offset offset_default{0, 0, 0, 0, 0, 0, 0};
    h5::stride stride_default{1, 1, 1, 1, 1, 1, 1};
    h5::block block_default{1, 1, 1, 1, 1, 1, 1};

    current_dims_default.rank = Rank;
    max_dims_default.rank = Rank;
    count_default.rank = Rank;
    offset_default.rank = Rank;
    stride_default.rank = Rank;
    block_default.rank = Rank;

    for (int i = 0; i < Rank; i++) {
        current_dims_default[i] = ref.dim(i);
        max_dims_default[i] = ref.dim(i);
        count_default[i] = ref.dim(i);
    }

    const h5::current_dims &current_dims = h5::arg::get(current_dims_default, args...);
    const h5::max_dims &max_dims = h5::arg::get(max_dims_default, args...);
    const h5::count &count = h5::arg::get(count_default, args...);
    const h5::offset &offset = h5::arg::get(offset_default, args...);
    const h5::stride &stride = h5::arg::get(stride_default, args...);
    const h5::block &block = h5::arg::get(block_default, args...);

    h5::ds_t ds;
    if (H5Lexists(fd, ref.name().c_str(), H5P_DEFAULT) <= 0) {
        ds = h5::create<T>(fd, ref.name(), current_dims, max_dims);
    } else {
        ds = h5::open(fd, ref.name());
    }
    h5::write(ds, ref, count, offset, stride);
}

template <typename T>
void write(const h5::fd_t &fd, const Tensor<T, 0> &ref) {
    h5::ds_t ds;
    if (H5Lexists(fd, ref.name().c_str(), H5P_DEFAULT) <= 0) {
        ds = h5::create<T>(fd, ref.name(), h5::current_dims{1});
    } else {
        ds = h5::open(fd, ref.name());
    }
    h5::write<T>(ds, ref.data(), h5::count{1});
}

template <size_t Rank, typename T, class... Args>
void write(const h5::fd_t &fd, const TensorView<T, Rank> &ref, Args &&...args) {
    h5::count count_default{1, 1, 1, 1, 1, 1, 1};
    h5::offset offset_default{0, 0, 0, 0, 0, 0, 0};
    h5::stride stride_default{1, 1, 1, 1, 1, 1, 1};
    h5::offset view_offset{0, 0, 0, 0, 0, 0, 0};
    h5::offset disk_offset{0, 0, 0, 0, 0, 0, 0};

    count_default.rank = Rank;
    offset_default.rank = Rank;
    stride_default.rank = Rank;
    view_offset.rank = Rank;
    disk_offset.rank = Rank;

    if (ref.stride(Rank - 1) != 1) {
        throw std::runtime_error("Final dimension of TensorView must be contiguous to write.");
    }

    const h5::offset &offset = h5::arg::get(offset_default, args...);
    const h5::stride &stride = h5::arg::get(stride_default, args...);

    count_default[Rank - 1] = ref.dim(Rank - 1);

    // Does the entry exist on disk?
    h5::ds_t ds;
    if (H5Lexists(state::data, ref.name().c_str(), H5P_DEFAULT) > 0) {
        ds = h5::open(fd, ref.name().c_str());
    } else {
        std::array<size_t, Rank> chunk_temp{};
        for (int i = 0; i < Rank; i++) {
            if (ref.dim(i) < 10)
                chunk_temp[i] = ref.dim(i);
            else
                chunk_temp[i] = 10;
        }

        ds = h5::create<T>(fd, ref.name().c_str(), h5::current_dims{ref.dims()},
                           h5::chunk{chunk_temp} | h5::gzip{9} | h5::fill_value<T>(0.0));
    }

    auto dims = get_dim_ranges<Rank - 1>(ref);

    for (auto combination : std::apply(ranges::views::cartesian_product, dims)) {
        // We generate all the cartesian products for all the dimensions except the final dimension
        // We call write on that final dimension.
        detail::add_elements<Rank - 1>(view_offset, offset_default, combination);
        detail::add_elements<Rank - 1>(disk_offset, offset, combination);

        // Get the data pointer from the view
        T *data = ref.data_array(view_offset);
        h5::write<T>(ds, data, count_default, disk_offset);
    }
}

// This needs to be expanded to handle the various h5 parameters like above.
template <size_t Rank, typename T>
auto read(const h5::fd_t &fd, const std::string &name) -> Tensor<T, Rank> {
    try {
        auto temp = h5::read<einsums::Tensor<T, Rank>>(fd, name);
        temp.set_name(name);
        return temp;
    } catch (std::exception &e) {
        println("Unable to open disk tensor '{}'", name);
        std::abort();
    }
}

template <typename T = double>
auto read(const h5::fd_t &fd, const std::string &name) -> Tensor<T, 0> {
    try {
        T temp{0};
        Tensor<T, 0> tensor{name};
        h5::read<T>(fd, name, &temp, h5::count{1});
        tensor = temp;
        return tensor;
    } catch (std::exception &e) {
        println("Unable to open disk tensor '{}'", name);
        std::abort();
    }
}

template <typename T, size_t Rank>
void zero(Tensor<T, Rank> &A) {
    A.zero();
}

template <typename T, size_t Rank>
struct DiskTensor final : public detail::TensorBase<T, Rank> {
    DiskTensor() = default;
    DiskTensor(const DiskTensor &) = default;
    DiskTensor(DiskTensor &&) noexcept = default;
    ~DiskTensor() = default;

    template <typename... Dims>
    explicit DiskTensor(h5::fd_t &file, std::string name, Dims... dims)
        : _file{file}, _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        struct stride {
            size_t value{1};
            stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), stride());

        std::array<size_t, Rank> chunk_temp{};
        for (int i = 0; i < Rank; i++) {
            if (_dims[i] < 10)
                chunk_temp[i] = _dims[i];
            else
                chunk_temp[i] = 64;
        }

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(_file, _name);
            } catch (std::exception &e) {
                println("Unable to open disk tensor '{}'", _name);
                std::abort();
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, h5::current_dims{static_cast<size_t>(dims)...},
                                      h5::chunk{chunk_temp} | h5::gzip{9} /* | h5::fill_value<T>(0.0) */);
            } catch (std::exception &e) {
                println("Unable to create disk tensor '{}'", _name);
                std::abort();
            }
        }
    }

    // Constructs a DiskTensor shaped like the provided Tensor. Data from the provided tensor
    // is NOT saved.
    explicit DiskTensor(h5::fd_t &file, const Tensor<T, Rank> &tensor) : _file{file}, _name{tensor.name()} {
        // Save dimension information from the provided tensor.
        h5::current_dims cdims;
        for (int i = 0; i < Rank; i++) {
            _dims[i] = tensor.dim(i);
            cdims[i] = _dims[i];
        }

        struct stride {
            size_t value{1};
            stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), stride());

        std::array<size_t, Rank> chunk_temp{};
        for (int i = 0; i < Rank; i++) {
            if (_dims[i] < 10)
                chunk_temp[i] = _dims[i];
            else
                chunk_temp[i] = 64;
        }

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(state::data, _name);
            } catch (std::exception &e) {
                println("Unable to open disk tensor '%s'", _name.c_str());
                std::abort();
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, cdims, h5::chunk{chunk_temp} | h5::gzip{9} | h5::fill_value<T>(0.0));
            } catch (std::exception &e) {
                println("Unable to create disk tensor '%s'", _name.c_str());
                std::abort();
            }
        }
    }

    // Provides ability to store another tensor to a part of a disk tensor.

    [[nodiscard]] auto dim(int d) const -> size_t { return _dims[d]; }
    auto dims() const -> Dim<Rank> { return _dims; }

    [[nodiscard]] auto existed() const -> bool { return _existed; }

    [[nodiscard]] auto disk() -> h5::ds_t & { return _disk; }

    // void _write(Tensor<T, Rank> &data) { h5::write(disk(), data); }

    [[nodiscard]] auto name() const -> const std::string & { return _name; }

    [[nodiscard]] auto stride(int d) const noexcept -> size_t { return _strides[d]; }

    // This creates a Disk object with its Rank being equal to the number of All{} parameters
    // Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
    auto operator()(MultiIndex... index)
        -> std::enable_if_t<count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0,
                            DiskView<T, count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>> {
        // Get positions of All
        auto all_positions =
            get_array_from_tuple<std::array<int, count_of_type<All_t, MultiIndex...>()>>(positions_of_type<All_t, MultiIndex...>());
        auto index_positions =
            get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(positions_of_type<size_t, MultiIndex...>());
        auto range_positions =
            get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(positions_of_type<Range, MultiIndex...>());

        const auto &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Stride<Rank> strides{};
        Count<Rank> counts{};

        std::fill(counts.begin(), counts.end(), 1.0);

        // Need the dim of the smaller tensor
        Dim<count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = get_from_tuple<size_t>(indices, value);
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            strides[value] = _strides[value];
            counts[value] = _dims[value];
            // dims_all[i] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = get_from_tuple<Range>(indices, value)[0];
            counts[value] = get_from_tuple<Range>(indices, value)[1] - get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (auto cnt : counts) {
            if (cnt > 1) {
                dims_all[dims_index++] = cnt;
            }
        }

        return DiskView<T, count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>(*this, dims_all, counts,
                                                                                                                offsets, strides);
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const
        -> std::enable_if_t<count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0,
                            const DiskView<T, count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>> {
        // Get positions of All
        auto all_positions =
            get_array_from_tuple<std::array<int, count_of_type<All_t, MultiIndex...>()>>(positions_of_type<All_t, MultiIndex...>());
        auto index_positions =
            get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(positions_of_type<size_t, MultiIndex...>());
        auto range_positions =
            get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(positions_of_type<Range, MultiIndex...>());

        const auto &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Stride<Rank> strides{};
        Count<Rank> counts{};

        std::fill(counts.begin(), counts.end(), 1.0);

        // Need the dim of the smaller tensor
        Dim<count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = get_from_tuple<size_t>(indices, value);
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            strides[value] = _strides[value];
            counts[value] = _dims[value];
            // dims_all[i] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = get_from_tuple<Range>(indices, value)[0];
            counts[value] = get_from_tuple<Range>(indices, value)[1] - get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (auto cnt : counts) {
            if (cnt > 1) {
                dims_all[dims_index++] = cnt;
            }
        }

        return DiskView<T, count_of_type<All_t, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>(*this, dims_all, counts,
                                                                                                                offsets, strides);
    }

  private:
    h5::fd_t &_file;

    std::string _name;
    Dim<Rank> _dims;
    Stride<Rank> _strides;

    h5::ds_t _disk;

    // Did the entry already exist on disk? Doesn't indicate validity of the data just the existance of the entry.
    bool _existed{false};
};

template <typename T, size_t ViewRank, size_t Rank>
struct DiskView final : public detail::TensorBase<T, ViewRank> {
    DiskView(DiskTensor<T, Rank> &parent, const Dim<ViewRank> &dims, const Count<Rank> &counts, const Offset<Rank> &offsets,
             const Stride<Rank> &strides)
        : _parent(parent), _dims(dims), _counts(counts), _offsets(offsets), _strides(strides), _tensor{_dims} {
        h5::read<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
    };
    DiskView(const DiskTensor<T, Rank> &parent, const Dim<ViewRank> &dims, const Count<Rank> &counts, const Offset<Rank> &offsets,
             const Stride<Rank> &strides)
        : _parent(const_cast<DiskTensor<T, Rank> &>(parent)), _dims(dims), _counts(counts), _offsets(offsets),
          _strides(strides), _tensor{_dims} {
        h5::read<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
        set_read_only(true);
    };
    DiskView(const DiskView &) = default;
    DiskView(DiskView &&) noexcept = default;

    ~DiskView() { put(); }

    void set_read_only(bool readOnly) { _readOnly = readOnly; }

    auto operator=(const T *other) -> DiskView & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        // Save the data to disk.
        h5::write<T>(_parent.disk(), other, h5::count{_counts}, h5::offset{_offsets});

        return *this;
    }

    // TODO: I'm not entirely sure if a TensorView can be sent to the disk.
    template <template <typename, size_t> typename TType>
    auto operator=(const TType<T, ViewRank> &other) -> DiskView & {
        if (_readOnly) {
            throw std::runtime_error("Attempting to write data to a read only disk view.");
        }

        // Check dims
        for (int i = 0; i < ViewRank; i++) {
            if (_dims[i] != other.dim(i)) {
                throw std::runtime_error(
                    fmt::format("DiskView::operator= : dims do not match (i {} dim {} other {})", i, _dims[i], other.dim(i)));
            }
        }

        // Sync the data to disk and into our internal tensor.
        h5::write<T>(_parent.disk(), other.data(), h5::count{_counts}, h5::offset{_offsets});
        _tensor = other;

        return *this;
    }

    void _create_tensor() {
        if (!_tensor) {
            _tensor = std::make_unique<Tensor<T, ViewRank>>(_dims);
        }
    }

    // Does not perform a disk read. That was handled by the constructor.
    auto get() -> Tensor<T, ViewRank> & { return _tensor; }

    void put() {
        if (!_readOnly)
            h5::write<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> const T & {
        return _tensor(std::forward<MultiIndex>(index)...);
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) -> T & {
        return _tensor(std::forward<MultiIndex>(index)...);
    }

    [[nodiscard]] auto dim(int d) const -> size_t { return _tensor.dim(d); }
    auto dims() const -> Dim<Rank> { return _tensor.dims(); }

    operator Tensor<T, ViewRank> &() const { return _tensor; }
    operator const Tensor<T, ViewRank> &() const { return _tensor; }

    void zero() { _tensor.zero(); }
    void set_all(T value) { _tensor.set_all(value); }

  private:
    DiskTensor<T, Rank> &_parent;
    Dim<ViewRank> _dims;
    Count<Rank> _counts;
    Offset<Rank> _offsets;
    Stride<Rank> _strides;
    Tensor<T, ViewRank> _tensor;

    bool _readOnly{false};

    // std::unique_ptr<Tensor<ViewRank, T>> _tensor;
};

#ifdef __cpp_deduction_guides
template <typename... Args>
Tensor(const std::string &, Args...) -> Tensor<double, sizeof...(Args)>;
template <typename T, size_t OtherRank, typename... Dims>
explicit Tensor(Tensor<T, OtherRank> &&otherTensor, std::string name, Dims... dims) -> Tensor<T, sizeof...(dims)>;
template <size_t Rank, typename... Args>
explicit Tensor(const Dim<Rank> &, Args...) -> Tensor<double, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(Tensor<T, OtherRank> &, const Dim<Rank> &, Args...) -> TensorView<T, Rank>;
template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(const Tensor<T, OtherRank> &, const Dim<Rank> &, Args...) -> TensorView<T, Rank>;
template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(TensorView<T, OtherRank> &, const Dim<Rank> &, Args...) -> TensorView<T, Rank>;
template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(const TensorView<T, OtherRank> &, const Dim<Rank> &, Args...) -> TensorView<T, Rank>;
template <typename T, size_t Rank, size_t OtherRank, typename... Args>
TensorView(std::string, Tensor<T, OtherRank> &, const Dim<Rank> &, Args...) -> TensorView<T, Rank>;

template <typename... Dims>
DiskTensor(h5::fd_t &file, std::string name, Dims... dims) -> DiskTensor<double, sizeof...(Dims)>;

// Supposedly C++20 will allow template deduction guides for template aliases. i.e. Dim, Stride, Offset, Count, Range.
#endif

// Useful factories
template <typename Type = double, typename... Args>
auto create_tensor(const std::string name, Args... args) -> Tensor<Type, sizeof...(Args)> {
    return Tensor<Type, sizeof...(Args)>{name, args...};
}

template <typename Type = double, typename... Args>
auto create_disk_tensor(h5::fd_t &file, const std::string name, Args... args) -> DiskTensor<Type, sizeof...(Args)> {
    return DiskTensor<Type, sizeof...(Args)>{file, name, args...};
}

template <typename T, size_t Rank>
auto create_disk_tensor_like(h5::fd_t &file, const Tensor<T, Rank> &tensor) -> DiskTensor<T, Rank> {
    return DiskTensor(file, tensor);
}

} // namespace einsums

struct TensorPrintOptions {
    int width{5};
    bool full_output{true};
};

template <template <typename, size_t> typename AType, size_t Rank, typename T>
auto println(const AType<T, Rank> &A, TensorPrintOptions options = {}) ->
    typename std::enable_if_t<std::is_base_of_v<einsums::detail::TensorBase<T, Rank>, AType<T, Rank>>> {
    println("Name: {}", A.name());
    {
        print::Indent indent{};

        if constexpr (einsums::is_incore_rank_tensor_v<AType<T, Rank>, Rank, T>) {
            if constexpr (std::is_same_v<AType<T, Rank>, einsums::Tensor<T, Rank>>)
                println("Type: In Core Tensor");
            else
                println("Type: In Core Tensor View");
        } else
            println("Type: Disk Tensor");

        println("Data Type: {}", type_name<T>());

        {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            println("Dims{{{}}}", oss.str().c_str());
        }

        {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.stride(i) << " ";
            }
            println("Strides{{{}}}", oss.str());
        }

        if (options.full_output) {
            println();

            if constexpr (Rank > 1 && einsums::is_incore_rank_tensor_v<AType<T, Rank>, Rank, T>) {
                auto target_dims = einsums::get_dim_ranges<Rank - 1>(A);
                auto final_dim = A.dim(Rank - 1);
                auto ndigits = einsums::ndigits(final_dim);

                for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
                    std::ostringstream oss;
                    for (int j = 0; j < final_dim; j++) {
                        if (j % options.width == 0) {
                            std::ostringstream tmp;
                            detail::TuplePrinterNoType<decltype(target_combination), Rank - 1>::print(tmp, target_combination);
                            if (final_dim >= j + options.width)
                                oss << fmt::format(
                                    "{:<14}", fmt::format("({}, {:{}d}-{:{}d}): ", tmp.str(), j, ndigits, j + options.width - 1, ndigits));
                            else
                                oss << fmt::format("{:<14}",
                                                   fmt::format("({}, {:{}d}-{:{}d}): ", tmp.str(), j, ndigits, final_dim - 1, ndigits));
                        }
                        auto new_tuple = std::tuple_cat(target_combination.base(), std::tuple(j));
                        T value = std::apply(A, new_tuple);
                        if (std::fabs(value) > 1.0E+10) {
                            if constexpr (std::is_floating_point_v<T>)
                                oss << "\x1b[0;37;41m" << fmt::format("{:14.8f} ", value) << "\x1b[0m";
                            else
                                oss << "\x1b[0;37;41m" << fmt::format("{:14d} ", value) << "\x1b[0m";
                        } else {
                            if constexpr (std::is_floating_point_v<T>)
                                oss << fmt::format("{:14.8f} ", value);
                            else
                                oss << fmt::format("{:14} ", value);
                        }
                        // } else {
                        // oss << std::setw(14) << 0.0;
                        // }
                        if (j % options.width == options.width - 1 && j != final_dim - 1) {
                            oss << "\n";
                        }
                    }
                    println("{}", oss.str());
                    println();
                }
            } else if constexpr (Rank == 1 && einsums::is_incore_rank_tensor_v<AType<T, Rank>, Rank, T>) {
                auto target_dims = einsums::get_dim_ranges<Rank>(A);

                for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
                    std::ostringstream oss;
                    oss << "(";
                    detail::TuplePrinterNoType<decltype(target_combination), Rank>::print(oss, target_combination);
                    oss << "): ";

                    T value = std::apply(A, target_combination);
                    if (std::fabs(value) > 1.0E+5) {
                        if constexpr (std::is_floating_point_v<T>)
                            oss << "\x1b[0;37;41m" << fmt::format("{:14.8f} ", value) << "\x1b[0m";
                        else
                            oss << "\x1b[0;37;41m" << fmt::format("{:14d} ", value) << "\x1b[0m";
                    } else {
                        if constexpr (std::is_floating_point_v<T>)
                            oss << fmt::format("{:14.8f} ", value);
                        else
                            oss << fmt::format("{:14d} ", value);
                    }

                    println("{}", oss.str());
                }
            }
        }
    }
    println();
}
