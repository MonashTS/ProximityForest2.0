cmake_minimum_required(VERSION 3.17)

project(Libtempo
        VERSION 0.0.1       # Todo: read from file
        DESCRIPTION "The tempo library"
        LANGUAGES CXX
        )

set(CMAKE_CXX_STANDARD 20)
include(cmake/project_options.cmake)


add_library(libtempo INTERFACE)
target_sources(libtempo
    INTERFACE
        ./include/libtempo/distance/cdtw.hpp
        ./include/libtempo/distance/distance.hpp
        ./include/libtempo/distance/dtw.hpp
        ./include/libtempo/distance/erp.hpp
        ./include/libtempo/distance/norm.hpp
        ./include/libtempo/distance/wdtw.hpp
        ./include/libtempo/tseries/tseries.hpp
        # UTILS
        ./include/libtempo/utils/capsule.hpp
        ./include/libtempo/utils/point.hpp
        ./include/libtempo/utils/utils.hpp
)
target_include_directories(libtempo INTERFACE ./include)


# CMake/IDE workaround: create a dummy target to force cmake/clion to consider header only library
add_executable(executable cmake/dummy.cpp)
target_link_libraries(executable PRIVATE libtempo)
target_link_libraries(executable PRIVATE project_options)
target_include_directories(executable PRIVATE ./include)



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Adding the tests
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_subdirectory(test)

