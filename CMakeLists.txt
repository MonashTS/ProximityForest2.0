cmake_minimum_required(VERSION 3.22)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Basic Settings
# Some of these settings are required to be set before the project call

# Include utility functions
include(cmake/cmake_utils.cmake)
detect_os()

# Set vcpkg settings
set(USE_VCPKG 1)

if (USE_VCPKG)
    if(WINDOWS)
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/${OS_NAME}/vcpkg/scripts/buildsystems/vcpkg.cmake")
#        set(CMAKE_TOOLCHAIN_FILE "E:/git/dotnet54/TS-CHIEF2/third_party/windows/vcpkg/scripts/buildsystems/vcpkg.cmake")
    elseif(LINUX OR OSX)
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/${OS_NAME}/vcpkg/scripts/buildsystems/vcpkg.cmake")
#        set(CMAKE_TOOLCHAIN_FILE "E:/git/dotnet54/TS-CHIEF2/third_party/x64-linux/vcpkg/scripts/buildsystems/vcpkg.cmake")
    else()
        message(FATAL_ERROR "Failed to setup vcpkg. Problem detecting OS, try resting CMake cache and then reloading.")
    endif()
    message(STATUS "Using vcpkg CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE}")
    message(STATUS "Using VCPKG_TARGET_TRIPLET ${VCPKG_TARGET_TRIPLET}")
endif()


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

print_env_info()

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Configurable options
option(BUILD_TESTING "Build tests." ON)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Compiler options

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # MESSAGE("Clang")
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


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Third party

# JSON CPP
if (USE_VCPKG)
    find_package(jsoncpp CONFIG REQUIRED)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(JSONCPP jsoncpp)
endif()

# MLPACK
if (USE_VCPKG)
    find_package(mlpack CONFIG REQUIRED)
    find_package(Armadillo CONFIG REQUIRED)
#    message(STATUS "MLPACK_INCLUDE_DIRS: ${MLPACK_INCLUDE_DIRS}")
#    message(STATUS "MLPACK_LIBRARIES: ${MLPACK_LIBRARIES}")
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
# this works without vcpkg
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP include: ${OpenMP_CXX_INCLUDE_DIRS}")
    message(STATUS "OpenMP library: ${OpenMP_CXX_LIBRARY}")
endif()



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Testing
# All the tests are compiled into one executable.
# Recursing into src/tempo will add the tests per component to the target created here
if(BUILD_TESTING)
    add_subdirectory(test/catch)
    add_executable(libtempo-test test/tests.cpp)
    target_include_directories(libtempo-test PRIVATE test)
    target_link_libraries(libtempo-test PRIVATE libtempo Catch2)
endif()


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create the target "libtempo"
add_subdirectory(src/tempo)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Add executable
add_subdirectory(exec)



#  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #  # Exec
  #  add_executable(testmlpack main.cpp)
  #  target_link_libraries(testmlpack PRIVATE ${MLPACK_LIBRARY})
  #  target_precompile_headers(testmlpack PRIVATE pch_std.hpp pch_lib.hpp)
  #  if(OpenMP_CXX_FOUND)
  #      target_link_libraries(testmlpack PRIVATE OpenMP::OpenMP_CXX)
  #  endif()
  #
  #  ### if (NOT BUILD_TESTS)
  #  ###    set_target_properties(exe1 exe2 PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
  #  ### endif()
  #
