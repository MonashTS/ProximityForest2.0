cmake_minimum_required(VERSION 3.22)
project(pch_test)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(third_party/catch)

add_subdirectory(third_party/armadillo-11.0.1)
get_target_property(_inc armadillo INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(armadillo SYSTEM INTERFACE ${_inc})
add_compile_definitions(armadillo INTERFACE $<$<CONFIG:RELEASE>:ARMA_NO_DEBUG>)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(third_party/json-3.10.5)

add_library(libtempo)
target_link_libraries(libtempo PUBLIC armadillo nlohmann_json::nlohmann_json m)
target_include_directories(libtempo PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)

add_executable(libtempo-test test/tests.cpp)
target_include_directories(libtempo-test PRIVATE test)
target_link_libraries(libtempo-test PRIVATE Catch2)
target_link_libraries(libtempo-test PRIVATE libtempo)

add_subdirectory(src/libtempo/classifier)
add_subdirectory(src/libtempo/distance)
add_subdirectory(src/libtempo/reader)
add_subdirectory(src/libtempo/transform)
add_subdirectory(src/libtempo/tseries)
add_subdirectory(src/libtempo/utils)

add_subdirectory(exec)
