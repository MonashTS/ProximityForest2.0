### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Add a new project
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
project(libtempo
        VERSION 0.0.1       # Todo: read from file
        DESCRIPTION "The tempo library"
        LANGUAGES CXX
        )


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Load our components
# Note: use EXCLUDE_FROM_ALL as we do not want to generate targets for subcomponents
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_subdirectory(classifier EXCLUDE_FROM_ALL)
add_subdirectory(distance EXCLUDE_FROM_ALL)
add_subdirectory(transform EXCLUDE_FROM_ALL)
add_subdirectory(tseries EXCLUDE_FROM_ALL)
add_subdirectory(reader EXCLUDE_FROM_ALL)
add_subdirectory(utils EXCLUDE_FROM_ALL)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# libtempo library
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_library(libtempo)
set_target_properties(libtempo PROPERTIES LINKER_LANGUAGE CXX)
target_sources(libtempo INTERFACE concepts.hpp)
target_link_libraries(libtempo INTERFACE libtempo::classifier)
target_link_libraries(libtempo INTERFACE libtempo::distance)
target_link_libraries(libtempo INTERFACE libtempo::transform)
#target_link_libraries(libtempo INTERFACE libtempo::tseries)
target_link_libraries(libtempo PUBLIC libtempo::reader)
target_link_libraries(libtempo INTERFACE libtempo::utils)




### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CMake/IDE workaround: create a dummy target to force cmake/clion to consider header only library
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_executable(executable _cmake/dummy.cpp)
target_link_libraries(executable PRIVATE libtempo)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Adding the tests
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Link with Catch, coming from our _test directory
add_executable(c2test _test/tests.cpp)
target_include_directories(c2test PRIVATE _test)
target_link_libraries(c2test PRIVATE Catch2)
target_link_libraries(c2test PRIVATE libtempo::test::distance)
target_link_libraries(c2test PRIVATE libtempo)