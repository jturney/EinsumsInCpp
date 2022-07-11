#pragma once

#include "einsums/Print.hpp"

#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>

namespace einsums {

namespace Arguments {

namespace Detail {

// declaration
template <class SearchPattern, int Position, int Count, bool Branch, class PrevHead, class Arguments>
struct TuplePosition;
// initialization case
template <class S, int P, int C, bool B, class not_used, class... Tail>
struct TuplePosition<S, P, C, B, not_used, std::tuple<Tail...>> : TuplePosition<S, P, C, false, not_used, std::tuple<Tail...>> {};
// recursive case
template <class S, int P, int C, class not_used, class Head, class... Tail>
struct TuplePosition<S, P, C, false, not_used, std::tuple<Head, Tail...>>
    : TuplePosition<S, P + 1, C, std::is_convertible<Head, S>::value, Head, std::tuple<Tail...>> {};
// match case
template <class S, int P, int C, class Type, class... Tail>
struct TuplePosition<S, P, C, true, Type, std::tuple<Tail...>> : std::integral_constant<int, P> {
    using type = Type;
    static constexpr bool present = true;
};
// default case
template <class S, class H, int P, int C>
struct TuplePosition<S, P, C, false, H, std::tuple<>> : std::integral_constant<int, -1> {
    static constexpr bool present = false;
};

} // namespace Detail

template <typename SearchPattern, typename... Args>
struct TuplePosition : Detail::TuplePosition<const SearchPattern &, -1, 0, false, void, std::tuple<Args...>> {};

template <typename SearchPattern, typename... Args,
          typename Idx = TuplePosition<const SearchPattern &, const Args &..., const SearchPattern &>>
auto get(const SearchPattern &definition, Args &&...args) -> typename Idx::type & {
    auto tuple = std::forward_as_tuple(args..., definition);
    return std::get<Idx::value>(tuple);
}

template <typename SearchPattern, typename... Args>
auto get(Args &&...args) -> SearchPattern & {
    auto tuple = std::forward_as_tuple(args...);
    return std::get<SearchPattern>(tuple);
}

template <int Idx, typename... Args>
auto getn(Args &&...args) -> typename std::tuple_element<Idx, std::tuple<Args...>>::type & {
    auto tuple = std::forward_as_tuple(args...);
    return std::get<Idx>(tuple);
}

// Does the parameter pack contain at least one of Type
template <typename T, typename... List>
struct Contains : std::true_type {};

template <typename T, typename Head, typename... Remaining>
struct Contains<T, Head, Remaining...> : std::conditional<std::is_same_v<T, Head>, std::true_type, Contains<T, Remaining...>>::type {};

template <typename T>
struct Contains<T> : std::false_type {};

} // namespace Arguments

/// Mimic Python's enumerate.
template <typename T, typename Iter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))> // The type of the end isn't needed but we must ensure
                                                            // it is valid.
constexpr auto enumerate(T &&iterable) {
    struct Iterator {
        std::size_t i;
        Iter iter;

        auto operator!=(const Iterator &other) const -> bool { return iter != other.iter; }
        void operator++() {
            ++i;
            ++iter;
        }
        auto operator*() const { return std::tie(i, *iter); }
    };
    struct IterableWrapper {
        T iterable;
        auto begin() { return Iterator{0, std::begin(iterable)}; }
        auto end() { return Iterator{0, std::end(iterable)}; }
    };

    return IterableWrapper{std::forward<T>(iterable)};
}

template <typename... T>
struct ZipRef {
  protected:
    std::tuple<T *...> ptr;

    template <std::size_t I = 0>
    void copy_assign(const ZipRef &z) {
        *(std::get<I>(ptr)) = *(std::get<I>(z.ptr));
        if constexpr (I + 1 < sizeof...(T))
            copy_assign<I + 1>(z);
    }

    template <std::size_t I = 0>
    void val_assign(const std::tuple<T...> &t) {
        *(std::get<I>(ptr)) = std::get<I>(t);
        if constexpr (I + 1 < sizeof...(T))
            val_assign<I + 1>(t);
    }

  public:
    ZipRef() = delete;
    ZipRef(const ZipRef &z) = default;
    ZipRef(ZipRef &&z) noexcept = default;
    ZipRef(T *const... p) : ptr(p...) {}

    auto operator=(const ZipRef &z) -> ZipRef & {
        copy_assign(z);
        return *this;
    }
    auto operator=(const std::tuple<T...> &val) -> ZipRef & {
        val_assign(val);
        return *this;
    }

    auto val() const -> std::tuple<T...> {
        return std::apply([](auto &&...args) { return std::tuple((*args)...); }, ptr);
    }
    operator std::tuple<T...>() const { return val(); }

    template <std::size_t I = 0>
    void swap_data(const ZipRef &o) const {
        std::swap(*std::get<I>(ptr), *std::get<I>(o.ptr));
        if constexpr (I + 1 < sizeof...(T))
            swap_data<I + 1>(o);
    }

    template <std::size_t N = 0>
    auto get() -> decltype(auto) {
        return *std::get<N>(ptr);
    }
    template <std::size_t N = 0>
    auto get() const -> decltype(auto) {
        return *std::get<N>(ptr);
    }

#define OPERATOR(OP)                                                                                                                       \
    bool operator OP(const ZipRef &o) const {                                                                                              \
        return val() OP o.val();                                                                                                           \
    }                                                                                                                                      \
    inline friend bool operator OP(const ZipRef &r, const std::tuple<T...> &t) {                                                           \
        return r.val() OP t;                                                                                                               \
    }                                                                                                                                      \
    inline friend bool operator OP(const std::tuple<T...> &t, const ZipRef &r) {                                                           \
        return t OP r.val();                                                                                                               \
    }

    OPERATOR(==) OPERATOR(<=) OPERATOR(>=) OPERATOR(!=) OPERATOR(<) OPERATOR(>)
#undef OPERATOR
};

template <typename... IT>
class ZipIter {
    std::tuple<IT...> it;

    template <std::size_t I = 0>
    auto one_is_equal(const ZipIter &rhs) const -> bool {
        if (std::get<I>(it) == std::get<I>(rhs.it))
            return true;
        if constexpr (I + 1 < sizeof...(IT))
            return one_is_equal<I + 1>(rhs);
        return false;
    }
    template <std::size_t I = 0>
    auto none_is_equal(const ZipIter &rhs) const -> bool {
        if (std::get<I>(it) == std::get<I>(rhs.it))
            return false;
        if constexpr (I + 1 < sizeof...(IT))
            return none_is_equal<I + 1>(rhs);
        return true;
    }

  public:
    using iterator_category = std::common_type_t<typename std::iterator_traits<IT>::iterator_category...>;
    using difference_type = std::common_type_t<typename std::iterator_traits<IT>::difference_type...>;
    using value_type = std::tuple<typename std::iterator_traits<IT>::value_type...>;
    using pointer = std::tuple<typename std::iterator_traits<IT>::pointer...>;
    using reference = ZipRef<std::remove_reference_t<typename std::iterator_traits<IT>::reference>...>;

    ZipIter() = default;
    ZipIter(const ZipIter &rhs) = default;
    ZipIter(ZipIter &&rhs) noexcept = default;
    ZipIter(const IT &...rhs) : it(rhs...) {}

    auto operator=(const ZipIter &rhs) -> ZipIter & = default;
    auto operator=(ZipIter &&rhs) noexcept -> ZipIter & = default;

    auto operator+=(const difference_type d) -> ZipIter & {
        std::apply([&d](auto &&...args) { ((std::advance(args, d)), ...); }, it);
        return *this;
    }
    auto operator-=(const difference_type d) -> ZipIter & { return operator+=(-d); }

    auto operator*() const -> reference {
        return std::apply([](auto &&...args) { return reference(&(*(args))...); }, it);
    }
    auto operator->() const -> pointer {
        return std::apply([](auto &&...args) { return pointer(&(*(args))...); }, it);
    }
    auto operator[](difference_type rhs) const -> reference { return *(operator+(rhs)); }

    auto operator++() -> ZipIter & { return operator+=(1); }
    auto operator--() -> ZipIter & { return operator+=(-1); }
    auto operator++(int) -> ZipIter {
        ZipIter tmp(*this);
        operator++();
        return tmp;
    }
    auto operator--(int) -> ZipIter {
        ZipIter tmp(*this);
        operator--();
        return tmp;
    }

    auto operator-(const ZipIter &rhs) const -> difference_type { return std::get<0>(it) - std::get<0>(rhs.it); }
    auto operator+(const difference_type d) const -> ZipIter {
        ZipIter tmp(*this);
        tmp += d;
        return tmp;
    }
    auto operator-(const difference_type d) const -> ZipIter {
        ZipIter tmp(*this);
        tmp -= d;
        return tmp;
    }
    inline friend auto operator+(const difference_type d, const ZipIter &z) -> ZipIter { return z + d; }
    inline friend auto operator-(const difference_type d, const ZipIter &z) -> ZipIter { return z - d; }

    // Since operator== and operator!= are often used to terminate cycles,
    // defining them as follow prevents incrementing behind the end() of a container
    auto operator==(const ZipIter &rhs) const -> bool { return one_is_equal(rhs); }
    auto operator!=(const ZipIter &rhs) const -> bool { return none_is_equal(rhs); }
#define OPERATOR(OP)                                                                                                                       \
    bool operator OP(const ZipIter &rhs) const {                                                                                           \
        return it OP rhs.it;                                                                                                               \
    }
    OPERATOR(<=) OPERATOR(>=) OPERATOR(<) OPERATOR(>)
#undef OPERATOR
};

template <typename... Container>
class Zip {
    std::tuple<Container &...> zip;

  public:
    Zip() = delete;
    Zip(const Zip &z) = default;
    Zip(Zip &&z) noexcept = default;
    Zip(Container &...z) : zip(z...) {}

#define HELPER(OP)                                                                                                                         \
    auto OP() {                                                                                                                            \
        return std::apply([](auto &&...args) { return ZipIter((args.OP())...); }, zip);                                                    \
    }                                                                                                                                      \
    auto c##OP() const {                                                                                                                   \
        return std::apply([](auto &&...args) { return ZipIter((args.c##OP())...); }, zip);                                                 \
    }                                                                                                                                      \
    auto OP() const {                                                                                                                      \
        return this->c##OP();                                                                                                              \
    }

    HELPER(begin) HELPER(end) HELPER(rbegin) HELPER(rend)
#undef HELPER
};

namespace Detail {

template <typename T, std::size_t... Is>
constexpr auto create_array(T value, std::index_sequence<Is...>) -> std::array<T, sizeof...(Is)> {
    // Cast Is to void to removing unused value warning
    return {{(static_cast<void>(Is), value)...}};
}

template <typename T, std::size_t... Is>
constexpr auto create_tuple(T value, std::index_sequence<Is...>) {
    return std::tuple{(static_cast<void>(Is), value)...};
}

template <typename T, std::size_t... Is>
constexpr auto create_tuple_from_array(const T &arr, std::index_sequence<Is...>) {
    return std::tuple((arr[Is])...);
}

template <typename T, int Position>
constexpr auto positions_of_type() {
    return std::make_tuple();
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto positions_of_type() {
    if constexpr (std::is_convertible_v<Head, T>) {
        return std::tuple_cat(std::make_tuple(Position), positions_of_type<T, Position + 1, Args...>());
    } else {
        return positions_of_type<T, Position + 1, Args...>();
    }
}
} // namespace Detail

template <typename T, typename... Args>
constexpr auto positions_of_type() {
    return Detail::positions_of_type<T, 0, Args...>();
}

template <typename T, typename... Args>
constexpr auto count_of_type(/*Args... args*/) {
    // return (std::is_same_v<Args, T> + ... + 0);
    return (std::is_convertible_v<Args, T> + ... + 0);
}

template <size_t N, typename T>
constexpr auto create_array(const T &value) -> std::array<T, N> {
    return Detail::create_array(value, std::make_index_sequence<N>());
}

template <size_t N, typename T>
constexpr auto create_tuple(const T &value) {
    return Detail::create_tuple(value, std::make_index_sequence<N>());
}

template <size_t N, typename T>
constexpr auto create_tuple_from_array(const T &arr) {
    return Detail::create_tuple_from_array(arr, std::make_index_sequence<N>());
}

template <typename Result, typename Tuple>
constexpr auto get_array_from_tuple(Tuple &&tuple) -> Result {
    constexpr auto get_array = [](auto &&...x) { return Result{std::forward<decltype(x)>(x)...}; };
    return std::apply(get_array, std::forward<Tuple>(tuple));
}

template <class Tuple, class F, std::size_t... I>
constexpr auto for_each_impl(Tuple &&t, F &&f, std::index_sequence<I...>) -> F {
    return (void)std::initializer_list<int>{(std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))), 0)...}, f;
}

template <class Tuple, class F>
constexpr auto for_each(Tuple &&t, F &&f) -> F {
    return for_each_impl(std::forward<Tuple>(t), std::forward<F>(f),
                         std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <typename ReturnType, typename Tuple>
inline auto get_from_tuple(Tuple &&tuple, size_t index) -> ReturnType {
    size_t currentIndex = 0;
    ReturnType returnValue{-1ul};

    for_each(tuple, [index, &currentIndex, &returnValue](auto &&value) {
        if (currentIndex == index) {
            // action(std::forward<decltype(value)>(value));
            if constexpr (std::is_convertible_v<ReturnType, std::remove_reference_t<decltype(value)>>)
                returnValue = value;
        }
        ++currentIndex;
    });
    return returnValue;
}

template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
    (static_cast<void>(f(std::integral_constant<T, S>{})), ...);
}

template <auto n, typename F>
constexpr void for_sequence(F f) {
    for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
}

namespace Detail {
template <typename T, typename Tuple>
struct HasType;

template <typename T>
struct HasType<T, std::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct HasType<T, std::tuple<U, Ts...>> : HasType<T, std::tuple<Ts...>> {};

template <typename T, typename... Ts>
struct HasType<T, std::tuple<T, Ts...>> : std::true_type {};
} // namespace Detail

template <typename S1, typename S2>
struct intersect {
    template <std::size_t... Indices>
    static auto make_intersection(std::index_sequence<Indices...>) {

        return std::tuple_cat(std::conditional_t<Detail::HasType<std::decay_t<std::tuple_element_t<Indices, S1>>, std::decay_t<S2>>::value,
                                                 std::tuple<std::tuple_element_t<Indices, S1>>, std::tuple<>>{}...);
    }
    using type = decltype(make_intersection(std::make_index_sequence<std::tuple_size<S1>::value>{}));
};

template <typename S1, typename S2>
using intersect_t = typename intersect<S1, S2>::type;

template <typename S1, typename S2>
struct difference {
    template <std::size_t... Indices>
    static auto make_difference(std::index_sequence<Indices...>) {

        return std::tuple_cat(std::conditional_t<Detail::HasType<std::decay_t<std::tuple_element_t<Indices, S1>>, std::decay_t<S2>>::value,
                                                 std::tuple<>, std::tuple<std::tuple_element_t<Indices, S1>>>{}...);
    }
    using type = decltype(make_difference(std::make_index_sequence<std::tuple_size<S1>::value>{}));
};

template <typename S1, typename S2>
using difference_t = typename difference<S1, S2>::type;

template <class Haystack, class Needle>
struct contains;

template <class Car, class... Cdr, class Needle>
struct contains<std::tuple<Car, Cdr...>, Needle> : contains<std::tuple<Cdr...>, Needle> {};

template <class... Cdr, class Needle>
struct contains<std::tuple<Needle, Cdr...>, Needle> : std::true_type {};

template <class Needle>
struct contains<std::tuple<>, Needle> : std::false_type {};

template <class Out, class In>
struct filter;

template <class... Out, class InCar, class... InCdr>
struct filter<std::tuple<Out...>, std::tuple<InCar, InCdr...>> {
    using type = typename std::conditional<contains<std::tuple<Out...>, InCar>::value,
                                           typename filter<std::tuple<Out...>, std::tuple<InCdr...>>::type,
                                           typename filter<std::tuple<Out..., InCar>, std::tuple<InCdr...>>::type>::type;
};

template <class Out>
struct filter<Out, std::tuple<>> {
    using type = Out;
};

template <class T>
using unique_t = typename filter<std::tuple<>, T>::type;

// template <class T>
// using c_unique_t = typename filter<std::tuple<>, const T>::type;
template <class T>
struct c_unique {
    using type = unique_t<std::decay_t<T>>;
};
template <class T>
using c_unique_t = typename c_unique<T>::type;

template <typename T, size_t Rank>
struct Tensor;

template <typename T, size_t Rank>
struct TensorView;

template <typename T, size_t Rank>
struct DiskTensor;

template <typename T, size_t ViewRank, size_t Rank>
struct DiskView;

template <typename D, size_t Rank, typename T>
struct is_incore_rank_tensor
    : public std::bool_constant<std::is_same_v<std::decay_t<D>, Tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, TensorView<T, Rank>>> {
};
template <typename D, size_t Rank, typename T>
inline constexpr bool is_incore_rank_tensor_v = is_incore_rank_tensor<D, Rank, T>::value;

template <typename D, size_t Rank, size_t ViewRank = Rank, typename T = double>
struct is_ondisk_tensor
    : public std::bool_constant<std::is_same_v<D, DiskTensor<T, Rank>> || std::is_same_v<D, DiskView<T, ViewRank, Rank>>> {};
template <typename D, size_t Rank, size_t ViewRank = Rank, typename T = double>
inline constexpr bool is_ondisk_tensor_v = is_ondisk_tensor<D, Rank, ViewRank, T>::value;

template <typename T>
struct is_smart_pointer_helper : public std::false_type {};

template <typename T>
struct is_smart_pointer_helper<std::shared_ptr<T>> : public std::true_type {};

template <typename T>
struct is_smart_pointer_helper<std::unique_ptr<T>> : public std::true_type {};

template <typename T>
struct is_smart_pointer_helper<std::weak_ptr<T>> : public std::true_type {};

template <typename T>
struct is_smart_pointer : public is_smart_pointer_helper<typename std::remove_cv<T>::type> {};

template <typename T>
inline constexpr bool is_smart_pointer_v = is_smart_pointer<T>::value;

template <typename T>
struct is_complex_t : public std::false_type {};

template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex_t<T>::value;

template <typename T>
struct complex_type {
    using type = T;
};

template <typename T>
struct complex_type<std::complex<T>> {
    using type = T;
};

template <typename T>
using complex_type_t = typename complex_type<T>::type;

template <typename T>
struct CircularBuffer {
    explicit CircularBuffer(size_t size) : _buffer(std::unique_ptr<T[]>(new T[size])), _max_size(size) {} // NOLINT

    void put(T item) {
        std::lock_guard<std::mutex> lock(_mutex);

        _buffer[_head] = item;

        if (_full) {
            _tail = (_tail + 1) % _max_size;
        }

        _head = (_head + 1) % _max_size;

        _full = _head == _tail;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(_mutex);
        _head = _tail;
        _full = false;
    }

    [[nodiscard]] auto empty() const -> bool { return (!_full && (_head == _tail)); }

    [[nodiscard]] auto full() const -> bool { return _full; }

    [[nodiscard]] auto capacity() const -> size_t { return _max_size; }

    [[nodiscard]] auto size() const -> size_t {
        size_t size = _max_size;

        if (!_full) {
            if (_head >= _tail) {
                size = _head - _tail;
            } else {
                size = _max_size + _head - _tail;
            }
        }

        return size;
    }

    auto operator[](int element) const -> const T & { return _buffer[element]; }

  private:
    std::unique_ptr<T[]> _buffer; // NOLINT
    const size_t _max_size;

    std::mutex _mutex;

    size_t _head{0};
    size_t _tail{0};
    bool _full{false};
};

namespace detail {
auto allocate_aligned_memory(size_t align, size_t size) -> void *;
void deallocate_aligned_memory(void *ptr) noexcept;
} // namespace detail

template <typename T, size_t Align = 32>
class AlignedAllocator;

template <size_t Align>
class AlignedAllocator<void, Align> {
  public:
    using pointer = void *;
    using const_pointer = const void *;
    using value_type = void;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };
};

template <typename T, size_t Align>
class AlignedAllocator {
  public:
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

  public:
    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

    [[nodiscard]] auto max_size() const noexcept -> size_type { return (size_type(~0) - size_type(Align)) / sizeof(T); }

    auto address(reference x) const noexcept -> pointer { return std::addressof(x); }

    auto address(const_reference x) const noexcept -> const_pointer { return std::addressof(x); }

    auto allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0) -> pointer {
        const auto alignment = static_cast<size_type>(Align);
        void *ptr = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept { return detail::deallocate_aligned_memory(p); }

    template <class U, class... Args>
    void construct(U *p, Args &&...args) {
        if constexpr (sizeof...(Args) > 0) {
            ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
        }
    }

    void destroy(pointer p) { p->~T(); }
};

template <typename T, size_t Align>
class AlignedAllocator<const T, Align> {
  public:
    using value_type = T;
    using pointer = const T *;
    using const_pointer = const T *;
    using reference = const T &;
    using const_reference = const T &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

  public:
    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

    [[nodiscard]] auto max_size() const noexcept -> size_type { return (size_type(~0) - size_type(Align)) / sizeof(T); }

    auto address(const_reference x) const noexcept -> const_pointer { return std::addressof(x); }

    auto allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0) -> pointer {
        const auto alignment = static_cast<size_type>(Align);
        void *ptr = detail::allocate_aligned_memory(alignment, n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept { return detail::deallocate_aligned_memory(p); }

    template <class U, class... Args>
    void construct(U *p, Args &&...args) {
        if constexpr (sizeof...(Args) > 0) {
            ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
        }
    }

    void destroy(pointer p) { p->~T(); }
};

template <typename T, size_t TAlign, typename U, size_t UAlign>
inline auto operator==(const AlignedAllocator<T, TAlign> &, const AlignedAllocator<U, UAlign> &) noexcept -> bool {
    return TAlign == UAlign;
}

template <typename T, size_t TAlign, typename U, size_t UAlign>
inline auto operator!=(const AlignedAllocator<T, TAlign> &, const AlignedAllocator<U, UAlign> &) noexcept -> bool {
    return TAlign != UAlign;
}

} // namespace einsums

namespace std {

template <std::size_t N, typename... T>
struct tuple_element<N, einsums::ZipRef<T...>> {
    using type = decltype(std::get<N>(std::declval<einsums::ZipRef<T...>>().val()));
};

template <typename... T>
struct tuple_size<einsums::ZipRef<T...>> : public std::integral_constant<std::size_t, sizeof...(T)> {};

template <std::size_t N, typename... T>
auto get(einsums::ZipRef<T...> &r) -> decltype(auto) {
    return r.template get<N>();
}
template <std::size_t N, typename... T>
auto get(const einsums::ZipRef<T...> &r) -> decltype(auto) {
    return r.template get<N>();
}

} // namespace std
