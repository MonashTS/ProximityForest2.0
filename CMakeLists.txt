cmake_minimum_required(VERSION 3.22)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Use ccache if available
# Must be before 'project', as the project command performs check on the compiler

find_program(CCACHE_EXECUTABLE ccache)
if(CCACHE_EXECUTABLE)
    message(STATUS "Using ccache ${CCACHE_EXECUTABLE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE PATH "ccache" FORCE)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE PATH "ccache" FORCE)
else()
    message("Consider installing ccache to speedup the build process")
endif()

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Overall project specification

project(tempo
        VERSION 0.0.1
        DESCRIPTION "Monash Time Series Classification Library"
        LANGUAGES CXX
        )


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(TOLOWER ${CMAKE_SYSTEM_NAME} OS_NAME)
message(STATUS "CMAKE_SYSTEM_NAME (CMAKE_SYSTEM): ${OS_NAME} (${CMAKE_SYSTEM})")

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Configurable options
option(BUILD_TESTING "Build tests." ON)

# modify here if using vcpkg
set(USE_VCPKG 0)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# VCPKG setup
if (USE_VCPKG)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/${OS_NAME}/vcpkg/scripts/buildsystems/vcpkg.cmake")

    if(NOT EXISTS ${CMAKE_TOOLCHAIN_FILE})
        message(FATAL_ERROR "Failed to load VCPKG from ${CMAKE_TOOLCHAIN_FILE}. Check if VCPKG is setup properly and if CMAKE_TOOLCHAIN_FILE path is correct.")
    endif()
    message(STATUS "Using VCPKG CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE}")
    message(STATUS "Using VCPKG_TARGET_TRIPLET ${VCPKG_TARGET_TRIPLET}")
endif()


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Compiler options
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # MESSAGE("Clang")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "AppleClang")
    # MESSAGE("AppleClang")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # MESSAGE("GNU")
    ### add_compile_options(-ftime-report)
    add_compile_options(-W -Wall)
    add_compile_options(-fconcepts-diagnostics-depth=2)
    add_compile_definitions(INTERFACE $<$<CONFIG:DEBUG>:_GLIBCXX_ASSERTIONS>)
    ### save-temps not usable with precompiled header?  add_compile_options(-fverbose-asm -save-temps)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # MESSAGE("Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # MESSAGE("MSVC")
    add_compile_options(/W3)
endif()

message(STATUS "=========================================================================================")
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Third party

# MLPACK
if (USE_VCPKG)
    find_package(mlpack CONFIG REQUIRED)
    find_package(Armadillo CONFIG REQUIRED)
    message(STATUS "MLPACK_INCLUDE_DIRS: ${MLPACK_INCLUDE_DIRS}")
    message(STATUS "MLPACK_LIBRARIES: ${MLPACK_LIBRARIES}")
else()
    # Put 'cmake' in the module path to get access to the 'FindMLPACK.cmake' file.
    # This file is provided by the mlpack project.
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(MLPACK REQUIRED)
    message(STATUS "MLPack include: ${MLPACK_INCLUDE_DIRS}")
    message(STATUS "MLPack library: ${MLPACK_LIBRARY}")

    # Armadillo: dependency of mlpack
    find_package(Armadillo REQUIRED)
endif()

# OPENMP
# This works without vcpkg
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP include: ${OpenMP_CXX_INCLUDE_DIRS}")
    message(STATUS "OpenMP library: ${OpenMP_CXX_LIBRARY}")
endif()

# RAPID CSV
add_library(rapidcsv INTERFACE)
target_include_directories(rapidcsv INTERFACE third_party/rapidcsv_mod)
target_sources(rapidcsv PUBLIC third_party/rapidcsv_mod/rapidcsv.h)

# NLOHMANN JSON
add_library(json INTERFACE bindings/python/src/pyPF/bindings.cpp bindings/python/src/pyPF/classifier/bindings.cpp bindings/python/src/pyPF/classifier/ProximityForest2/pybind_pf2.hpp)
target_include_directories(json INTERFACE third_party/nlohmann/include)
target_sources(json PUBLIC third_party/nlohmann/include/nlohmann/json.hpp)

# Command line parser
add_subdirectory(third_party/cxxopts)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Testing
# All the tests are compiled into one executable.
# Recursing into src/tempo will add the tests per component to the target created here
if(BUILD_TESTING)
    add_subdirectory(test/Catch2)
    add_executable(libtempo-test test/tests.cpp)
    target_include_directories(libtempo-test PRIVATE test)
    target_link_libraries(libtempo-test PRIVATE libtempo Catch2::Catch2WithMain)
endif()


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create the target "libtempo"
add_subdirectory(src/tempo)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Add executable
add_subdirectory(exec)

