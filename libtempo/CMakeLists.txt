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
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_subdirectory(distance)
add_subdirectory(tseries)
add_subdirectory(reader)
add_subdirectory(utils)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# libtempo library
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_library(libtempo INTERFACE)
target_link_libraries(libtempo INTERFACE libtempo::distance)
target_link_libraries(libtempo INTERFACE libtempo::tseries)
target_link_libraries(libtempo INTERFACE libtempo::reader)
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
add_subdirectory(_test)
add_executable(c2test _test/tests.cpp)
target_link_libraries(c2test PRIVATE Catch)
target_link_libraries(c2test PRIVATE libtempo::test::distance)
target_link_libraries(c2test PRIVATE libtempo)

#
# target_sources(test_distuni
#     PRIVATE
#         tests.cpp
#         mock/mockseries.hpp
#         # --- Libtempo distances - Univariate
#         distance/univariate/dtw.cpp
#         distance/univariate/adtw.cpp
#         distance/univariate/cdtw.cpp
#         distance/univariate/wdtw.cpp
#         distance/univariate/erp.cpp
#         distance/univariate/direct.cpp
#         distance/univariate/msm.cpp
#         distance/univariate/twe.cpp
#         distance/univariate/lcss.cpp
#         # --- Libtempo distances - Multivariate
#         distance/multivariate_dep/m_dtw.cpp
#         distance/multivariate_dep/m_cdtw.cpp
#         distance/multivariate_dep/m_wdtw.cpp
#         distance/multivariate_dep/m_erp.cpp
#         distance/multivariate_dep/m_direct.cpp
#         distance/multivariate_dep/m_msm.cpp
#         distance/multivariate_dep/m_twe.cpp
#         distance/multivariate_dep/m_lcss.cpp
# )
#
# target_link_libraries(test_distuni PRIVATE project_options)
# target_link_libraries(test_distuni PRIVATE Catch)
# target_link_libraries(test_distuni PRIVATE libtempo)


