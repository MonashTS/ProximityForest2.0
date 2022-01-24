cmake_minimum_required(VERSION 3.17)
project(tempo
        VERSION 0.0.1
        DESCRIPTION "Time Series classification library"
        LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# C++ and Interface target
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 3.4 REQUIRED NO_MODULE PATHS ${CMAKE_CURRENT_LIST_FILE}/third_party/Eigen NO_DEFAULT_PATHS)

add_subdirectory(libtempo)