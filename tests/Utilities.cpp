#include "einsums/Utilities.hpp"

#include "einsums/Tensor.hpp"

#include <catch2/catch.hpp>

TEST_CASE("arange") {
    using namespace einsums;

    SECTION("stop") {
        // Since 10 by default is an int the tensor returned is also an int.
        auto tensor = arange(10);

        // println(tensor);
    }

    SECTION("start-stop") {
        auto tensor = arange<double>(5, 20, 5);
        // println(tensor);
    }
}