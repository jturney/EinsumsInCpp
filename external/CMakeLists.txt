set(RANGES_CXX_STD 17)
add_subdirectory(range-v3-0.11.0)
add_subdirectory(fmt-8.1.1)
add_subdirectory(catch-2.13.3)
add_subdirectory(h5cpp-1.10.4-6-EAT)

if(EINSUMS_USE_HPTT)
	add_subdirectory(hptt)
endif()

set(BUILD_FUNCTIONAL_TESTS OFF CACHE INTERNAL "")
include(FetchContent)
FetchContent_Declare(
	oneMKL
	GIT_REPOSITORY https://github.com/oneapi-src/oneMKL.git
	GIT_TAG 7c24bc66da834a39932cfead5ac66f54c729a0a2 # Master as of 3/25/2022
)
FetchContent_MakeAvailable(oneMKL)
