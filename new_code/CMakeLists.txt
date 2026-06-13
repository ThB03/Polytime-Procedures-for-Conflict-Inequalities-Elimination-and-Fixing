# =============================================================================
#  CMakeLists.txt  --  build configuration for the implication-graph
#  presolver plugin and its companion binaries.
#
#  Targets:
#    graph_utils         (static library, no SCIP)
#    test_graph_utils    (unit-test binary, depends on graph_utils)
#    standalone_runner   (no-SCIP arc-list runner, depends on graph_utils)
#    scip_implgraph      (full SCIP driver with the plugin linked in;
#                         built only when SCIP is found)
#
#  Out-of-source build (recommended):
#    mkdir build && cd build
#    cmake .. [-DSCIP_DIR=/path/to/scip/install/lib/cmake/scip]
#    cmake --build . -j
#    ctest --output-on-failure          # runs test_graph_utils
#
#  If SCIP is not installed, only the standalone targets are built; the unit
#  tests still pass and the standalone runner can validate the algorithmic core
#  against the Python reference (see scripts/README.md).
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(implgraph_presolver CXX)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -O2 already gives Tarjan + bitset DP within 10% of -O3 in our microbenches;
# we prefer -O2 to keep debug symbols readable in case the user wants to
# attach a profiler.
add_compile_options(-Wall -Wextra -Wno-sign-compare)

# -----------------------------------------------------------------------------
#  CUDA -- optional GPU backend for the reachability pass
# -----------------------------------------------------------------------------
# Detect CUDA the cmake-3.18+ way (FindCUDA is deprecated).  If CUDA is found,
# we enable CXX <-> CUDA cross-compilation and build graph_utils_gpu.cu plus
# the bench_reach_gpu standalone benchmark.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # CUDA_ARCHITECTURES: target the user's actual GPU.  Blackwell consumer /
    # workstation parts (RTX 50xx and RTX PRO 5000/6000) are sm_120; Hopper
    # (H100) is sm_90; Ada (RTX 40xx) is sm_89; Ampere (A100) is sm_80;
    # datacenter Blackwell (B100/B200) is sm_100.
    #
    # IMPORTANT: do NOT use "native" -- nvcc 13's native detection picks
    # sm_75 (Turing) on WSL2 for reasons unrelated to the installed device,
    # which causes the driver to PTX-JIT every kernel launch at ~150 ms cost.
    # We force sm_120 by default; override with
    #     -DCMAKE_CUDA_ARCHITECTURES="80;89;120"
    # on the cmake line if you need fat binaries for multiple devices.
    #
    # Setting both the CACHE variable AND the plain variable BEFORE
    # enable_language(CUDA) ensures CMake bakes the choice into the per-target
    # default; otherwise enable_language() may stamp the target with its own
    # auto-detected value and our later setting is ignored.
    set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING
        "CUDA compute capabilities to compile for (semicolon-separated)" FORCE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(IMPLGRAPH_HAS_CUDA ON)
    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}); "
                   "building GPU reachability backend")
else()
    set(IMPLGRAPH_HAS_CUDA OFF)
    message(STATUS "CUDA not found; GPU reachability backend disabled. "
                   "Install the CUDA toolkit and rerun cmake to enable.")
endif()

# -----------------------------------------------------------------------------
#  Core: graph_utils (no SCIP)
# -----------------------------------------------------------------------------
add_library(graph_utils STATIC src/graph_utils.cpp)
target_include_directories(graph_utils PUBLIC src)

# -----------------------------------------------------------------------------
#  GPU backend (no SCIP) -- conditionally compiled when CUDA is available
# -----------------------------------------------------------------------------
if(IMPLGRAPH_HAS_CUDA)
    add_library(graph_utils_gpu STATIC src/graph_utils_gpu.cu)
    target_include_directories(graph_utils_gpu PUBLIC src)
    target_link_libraries(graph_utils_gpu PUBLIC graph_utils)
    # The .cu file links to libcudart implicitly via nvcc; explicit on shared
    # libraries on some Linux toolchains.  PRIVATE because consumers don't need
    # the CUDA runtime in their own object files.
    target_link_libraries(graph_utils_gpu PRIVATE cudart)
    set_target_properties(graph_utils_gpu PROPERTIES
        # CUDA_SEPARABLE_COMPILATION OFF (whole-program compilation): our
        # kernel is self-contained -- no cross-translation-unit __device__
        # references -- so we don't need separable compilation, and turning
        # it off avoids the device-link symbol mismatch that hits the
        # downstream scip_implgraph executable.
        CUDA_SEPARABLE_COMPILATION  OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE   ON
        # Explicit per-target arch list; overrides anything enable_language()
        # stamped as the per-target default.  Confirm with:
        #   cuobjdump --list-elf libgraph_utils_gpu.a
        CUDA_ARCHITECTURES          "120")

    # Standalone CPU-vs-GPU benchmark + correctness driver.  Builds even
    # without SCIP; useful for validating the GPU kernel in isolation.
    add_executable(bench_reach_gpu src/bench_reach_gpu.cpp)
    target_link_libraries(bench_reach_gpu PRIVATE graph_utils graph_utils_gpu)
endif()

# -----------------------------------------------------------------------------
#  Unit tests
# -----------------------------------------------------------------------------
enable_testing()
add_executable(test_graph_utils test/test_graph_utils.cpp)
target_link_libraries(test_graph_utils PRIVATE graph_utils)
add_test(NAME graph_utils_unit COMMAND test_graph_utils)

# -----------------------------------------------------------------------------
#  Standalone arc-list runner (no SCIP)
# -----------------------------------------------------------------------------
add_executable(standalone_runner src/standalone_runner.cpp)
target_link_libraries(standalone_runner PRIVATE graph_utils)

# -----------------------------------------------------------------------------
#  SCIP driver -- optional
# -----------------------------------------------------------------------------
#
# We support both a `find_package(SCIP)` install (the recommended path -- SCIP
# 9+ ships SCIPConfig.cmake) and the older raw header/library setup via
# SCIP_INCLUDE_DIR / SCIP_LIBRARY.
#
find_package(SCIP QUIET CONFIG)
if(NOT SCIP_FOUND)
    if(DEFINED ENV{SCIP_DIR})
        find_package(SCIP QUIET CONFIG PATHS "$ENV{SCIP_DIR}/lib/cmake/scip")
    endif()
endif()

if(SCIP_FOUND)
    message(STATUS "SCIP found (version ${SCIP_VERSION}); building scip_implgraph")

    # ----- COIN-OR / CBC dependency for the CglProbing-based extractor -----
    #
    # Ubuntu ships these as coinor-libcbc-dev / coinor-libcgl-dev /
    # coinor-libosi-dev / coinor-libclp-dev / coinor-libcoinutils-dev.
    # They do NOT ship CMake config files, so we locate them manually.
    find_path(COIN_INCLUDE_DIR
        NAMES CglProbing.hpp
        PATH_SUFFIXES coin
        DOC "Path containing CBC headers (CglProbing.hpp, OsiClpSolverInterface.hpp, ...)")
    find_library(CBC_LIB        NAMES Cbc)
    find_library(CGL_LIB        NAMES Cgl)
    find_library(OSI_LIB        NAMES Osi)
    find_library(OSICLP_LIB     NAMES OsiClp)
    find_library(CLP_LIB        NAMES Clp)
    find_library(COINUTILS_LIB  NAMES CoinUtils)

    set(CBC_ALL_FOUND TRUE)
    foreach(_lib COIN_INCLUDE_DIR CBC_LIB CGL_LIB OSI_LIB OSICLP_LIB CLP_LIB COINUTILS_LIB)
        if(NOT ${_lib})
            message(WARNING "CBC dependency not found: ${_lib}")
            set(CBC_ALL_FOUND FALSE)
        endif()
    endforeach()

    add_library(presol_implgraph STATIC src/presol_implgraph.cpp)
    target_link_libraries(presol_implgraph PUBLIC graph_utils ${SCIP_LIBRARIES})
    target_include_directories(presol_implgraph PUBLIC ${SCIP_INCLUDE_DIRS})

    # Link the GPU backend so the plug-in can dispatch to it for large
    # condensations when presolving/implgraph/usegpu != 0.
    if(IMPLGRAPH_HAS_CUDA)
        target_link_libraries(presol_implgraph PUBLIC graph_utils_gpu)
        target_compile_definitions(presol_implgraph PUBLIC IMPLGRAPH_HAS_CUDA=1)
    endif()

    if(CBC_ALL_FOUND)
        message(STATUS "CBC found at ${COIN_INCLUDE_DIR}; building with CglProbing extractor")
        target_sources           (presol_implgraph PRIVATE src/cbc_extractor.cpp)
        target_compile_definitions(presol_implgraph PUBLIC IMPLGRAPH_HAS_CBC=1)
        target_include_directories(presol_implgraph PUBLIC ${COIN_INCLUDE_DIR})
        target_link_libraries(presol_implgraph PUBLIC
            ${CGL_LIB} ${CBC_LIB} ${OSICLP_LIB} ${OSI_LIB} ${CLP_LIB} ${COINUTILS_LIB})
    else()
        message(WARNING "CBC not found; presol_implgraph will build without "
            "CglProbing -- only the SCIP constraint-handler extractor will be available.")
    endif()

    add_executable(scip_implgraph src/cmain.cpp)
    target_link_libraries(scip_implgraph PRIVATE presol_implgraph)

    # GPU-native conflict graph builder (replaces CBC for large instances).
    # Requires SCIP (for MPS parsing) AND CUDA (for the extractor kernel).
    if(IMPLGRAPH_HAS_CUDA)
        add_library(gpu_conflict_extractor STATIC src/gpu_conflict_extractor.cu)
        target_include_directories(gpu_conflict_extractor PUBLIC src)
        set_target_properties(gpu_conflict_extractor PROPERTIES
            CUDA_SEPARABLE_COMPILATION  OFF
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE   ON
            CUDA_ARCHITECTURES          "120")
        target_link_libraries(gpu_conflict_extractor PRIVATE cudart)

        add_executable(gpu_conflict_graph src/gpu_conflict_main.cpp)
        target_link_libraries(gpu_conflict_graph PRIVATE
            gpu_conflict_extractor ${SCIP_LIBRARIES})
        target_include_directories(gpu_conflict_graph PRIVATE ${SCIP_INCLUDE_DIRS})
    endif()
else()
    message(WARNING
        "SCIP not found.  Building only the standalone targets.\n"
        "  Set -DSCIP_DIR=<scip-install>/lib/cmake/scip on the cmake line\n"
        "  or set the SCIP_DIR environment variable to enable the SCIP driver.")
endif()
