add_executable(test-all
    Common/main.cpp
    Disk.cpp
    LinearAlgebra.cpp
    Tensor.cpp
    TensorAlgebra.cpp
    Utilities.cpp
)

if (HAVE_MKL_LAPACKE_HEADER OR LAPACKE_FOUND)
    target_sources(test-all PUBLIC Decomposition.cpp)
endif()

target_link_libraries(test-all Catch2::Catch2 einsums)
