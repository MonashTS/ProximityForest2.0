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
        # --- Libtempo distances - Multivariate
        distance/multivariate_dep/m_dtw.cpp
        distance/multivariate_dep/m_cdtw.cpp
        # --- Reference implementation of distances
        distance/univariate/references/dtw/dtw.cpp
        distance/univariate/references/dtw/cdtw.cpp
        distance/univariate/references/dtw/wdtw.cpp
        distance/univariate/references/erp/erp.cpp
        distance/univariate/references/lcss/lcss.cpp
        distance/univariate/references/msm/msm.cpp
        distance/univariate/references/twe/twe.cpp
)

target_link_libraries(test_distuni PRIVATE project_options)
target_link_libraries(test_distuni PRIVATE Catch)
target_link_libraries(test_distuni PRIVATE libtempo)
