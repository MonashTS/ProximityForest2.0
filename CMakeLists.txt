### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)

find_program(CCACHE_EXECUTABLE ccache)
if(CCACHE_EXECUTABLE)
    message(STATUS "use ccache ${CCACHE_EXECUTABLE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE PATH "ccache" FORCE)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE PATH "ccache" FORCE)
endif()

project(tempo
        VERSION 0.0.1
        DESCRIPTION "Monash Time Series Classification Library"
        LANGUAGES CXX
        )



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
# Third parties

# Catch test framework
add_subdirectory(third_party/catch)

# Armadillo
add_subdirectory(third_party/armadillo-11.0.1)
get_target_property(_inc armadillo INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(armadillo SYSTEM INTERFACE ${_inc})
add_compile_definitions(armadillo INTERFACE $<$<CONFIG:RELEASE>:ARMA_NO_DEBUG>)

# JsonCPP
add_subdirectory(third_party/jsoncpp-1.9.5)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Our targets

# Libtempo
add_library(libtempo)
target_link_libraries(libtempo PUBLIC armadillo jsoncpp)
target_include_directories(libtempo PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)

# Tests
add_executable(libtempo-test test/tests.cpp)
target_include_directories(libtempo-test PRIVATE test)
target_link_libraries(libtempo-test PRIVATE Catch2 libtempo)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Collects source and tests
add_subdirectory(src/libtempo/classifier)
add_subdirectory(src/libtempo/distance)
add_subdirectory(src/libtempo/reader)
add_subdirectory(src/libtempo/transform)
add_subdirectory(src/libtempo/tseries)
add_subdirectory(src/libtempo/utils)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Add executables, depends on the tempo library
add_subdirectory(exec)
