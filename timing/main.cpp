#include "einsums/Blas.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/State.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/Timer.hpp"
#include "einsums/Utilities.hpp"

#include <cstdlib>

auto main() -> int {
    ////////////////////////////////////
    // Form the two-electron integrals//
    ////////////////////////////////////
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    timer::initialize();
    blas::initialize();

    // Disable HDF5 diagnostic reporting.
    H5Eset_auto(0, nullptr, nullptr);

    // Create a file to hold the data from the DiskTensor tests.
    einsums::state::data = h5::create("Data.h5", H5F_ACC_TRUNC);

#define NMO 64
#define NBS 200

    int nmo1{NMO}, nmo2{NMO}, nmo3{NMO}, nmo4{NMO};
    int nbs1{NBS}, nbs2{NBS}, nbs3{NBS}, nbs4{NBS};

    println("Running on {} threads", omp_get_max_threads());

#if 0
    println("NMO {} :: NBS {}", NMO, NBS);

    timer::push("Allocations");
    auto GAO = std::make_unique<Tensor<4>>("AOs", nbs1, nbs2, nbs3, nbs4);
    Tensor<2> C1{"C1", nbs1, nmo1};
    Tensor<2> C2{"C2", nbs2, nmo2};
    Tensor<2> C3{"C3", nbs3, nmo3};
    Tensor<2> C4{"C4", nbs4, nmo4};
    timer::pop();

    timer::push("Full Transformation");

    // Transform ERI AO Tensor to ERI MO Tensor
    timer::push("C4");
    timer::push("Allocation 1");
    auto pqrS = std::make_unique<Tensor<4>>("pqrS", nbs1, nbs2, nbs3, nmo4);
    timer::pop();
    einsum(Indices{p, q, r, S}, &pqrS, Indices{p, q, r, s}, GAO, Indices{s, S}, C4);
    GAO.reset(nullptr);
    timer::pop();

    timer::push("C3");
    timer::push("Allocation 1");
    auto pqSr = std::make_unique<Tensor<4>>("pqSr", nbs1, nbs2, nmo4, nbs3);
    timer::pop();
    timer::push("presort");
    sort(Indices{p, q, S, r}, &pqSr, Indices{p, q, r, S}, pqrS);
    timer::pop();
    pqrS.reset(nullptr);

    timer::push("Allocation 2");
    auto pqSR = std::make_unique<Tensor<4>>("pqSR", nbs1, nbs2, nmo4, nmo3);
    timer::pop();
    einsum(Indices{p, q, S, R}, &pqSR, Indices{p, q, S, r}, pqSr, Indices{r, R}, C3);
    pqSr.reset(nullptr);
    timer::pop();

    timer::push("C2");
    timer::push("Allocation 1");
    auto RSpq = std::make_unique<Tensor<4>>("RSpq", nmo3, nmo4, nbs1, nbs2);
    timer::pop();
    timer::push("presort");
    sort(Indices{R, S, p, q}, &RSpq, Indices{p, q, S, R}, pqSR);
    pqSR.reset(nullptr);
    timer::pop();

    timer::push("Allocation 2");
    auto RSpQ = std::make_unique<Tensor<4>>("RSpQ", nmo3, nmo4, nbs1, nmo2);
    timer::pop();
    einsum(Indices{R, S, p, Q}, &RSpQ, Indices{R, S, p, q}, RSpq, Indices{q, Q}, C2);
    RSpq.reset(nullptr);
    timer::pop();

    timer::push("C1");
    timer::push("Allocation 1");
    auto RSQp = std::make_unique<Tensor<4>>("RSQp", nmo3, nmo4, nmo2, nbs1);
    timer::pop();
    timer::push("presort");
    sort(Indices{R, S, Q, p}, &RSQp, Indices{R, S, p, Q}, RSpQ);
    RSpQ.reset(nullptr);
    timer::pop();

    timer::push("Allocation 2");
    auto RSQP = std::make_unique<Tensor<4>>("RSQP", nmo3, nmo4, nmo2, nmo1);
    timer::pop();
    einsum(Indices{R, S, Q, P}, &RSQP, Indices{R, S, Q, p}, RSQp, Indices{p, P}, C1);
    RSQp.reset(nullptr);
    timer::pop();

    timer::push("Sort RSQP -> PQRS");
    timer::push("Allocation");
    Tensor<4> PQRS{"PQRS", nmo1, nmo2, nmo3, nmo4};
    timer::pop();
    sort(Indices{P, Q, R, S}, &PQRS, Indices{R, S, Q, P}, RSQP);
    RSQP.reset(nullptr);
    timer::pop();

    timer::pop(); // Full Transformation

    element_transform(&PQRS, [](double value) -> double { return 1.0 / value; });
#endif

    // const size_t size = 7;
    // const size_t d1 = 4;

    // Tensor<3> I_original = create_random_tensor("Original", size, size, size);
    // println(I_original);

    // TensorView<2> I_view = I_original(d1, All, All);
    // println(I_view);

    // for (size_t i = 0; i < size; i++) {
    //     for (size_t j = 0; j < size; j++) {
    //         I_original(d1, i, j) == I_view(i, j);
    //     }
    // }

    // size_t _i = 3, _j = 4, _k = 5;

    // Tensor<3> A = create_random_tensor("A", _i, _j, _i);
    // Tensor<3> B = create_random_tensor("B", _j, _i, _j);
    // Tensor<3> C{"Einsum C", _i, _j, _i};
    // Tensor<3> C0{"Correct C", _i, _j, _i};
    // C0.zero();
    // C.zero();

    // for (size_t i0 = 0; i0 < _i; i0++) {
    //     for (size_t j0 = 0; j0 < _j; j0++) {
    //         println("C({}, {}, {}) A({}, {}, {}) B({}, {}, {})", i0, j0, i0, i0, j0, i0, j0, i0, j0);
    //         C0(i0, j0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
    //     }
    // }

    // einsum(Indices{i, j, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B);

    // println(C0);
    // println(C);

    // for (size_t i0 = 0; i0 < _i; i0++) {
    //     for (size_t j0 = 0; j0 < _j; j0++) {
    //         CHECK_THAT(C(i0, j0, i0), Catch::Matchers::WithinRel(C0(i0, j0, i0), 0.00001));
    //     }
    // }

    // Tensor<2> A = create_random_tensor("A", 3, 3);
    // Tensor<1> B = create_random_tensor("B", 3);
    // Tensor<3> C{"C", 3, 3, 3};

    // C.set_all(0.0);
    // einsum(Indices{i, j, k}, &C, Indices{i, j}, A, Indices{k}, B);

    // auto eri = create_random_tensor("eri", NMO, NMO, NMO, NMO);
    // DiskTensor g(state::data, "eri", NMO, NMO, NMO, NMO);

    // {
    //     Section section{"disk write"};
    //     g(All, All, All, All) = eri;
    // }

    auto a = create_incremented_tensor("a", 3, 5);
    println(a);

    timer::report();
    blas::finalize();
    timer::finalize();

    // Typically you would build a new wavefunction and populate it with data
    return EXIT_SUCCESS;
}
