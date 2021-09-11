### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Setting up the test library catch2
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_library(Catch INTERFACE)
target_sources(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/catch)

add_definitions("-fverbose-asm")

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Univariate distance
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_executable(test_distuni)

target_sources(test_distuni
    PRIVATE
        tests.cpp
        mock/mockseries.hpp
        # --- Libtempo distances - Univariate
        distance/univariate/dtw.cpp
        distance/univariate/cdtw.cpp
        distance/univariate/wdtw.cpp
        distance/univariate/erp.cpp
        distance/univariate/norm.cpp
        distance/univariate/msm.cpp
        distance/univariate/twe.cpp
        # --- Libtempo distances - Multivariate
        distance/multivariate_dep/m_dtw.cpp
        distance/multivariate_dep/m_cdtw.cpp
        distance/multivariate_dep/m_wdtw.cpp
        distance/multivariate_dep/m_erp.cpp
        distance/multivariate_dep/m_norm.cpp
        distance/multivariate_dep/m_msm.cpp
        distance/multivariate_dep/m_twe.cpp
)

target_link_libraries(test_distuni PRIVATE project_options)
target_link_libraries(test_distuni PRIVATE Catch)
target_link_libraries(test_distuni PRIVATE libtempo)
