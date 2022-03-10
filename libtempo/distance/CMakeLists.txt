### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::distance component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_library(libtempo-distance INTERFACE
        cdtw.hpp
        adtw.hpp
        direct.hpp
        cost_function.hpp
        dtw.hpp
        erp.hpp
        lcss.hpp
        msm.hpp
        twe.hpp
        wdtw.hpp
        )

target_include_directories(libtempo-distance INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_library(libtempo::distance ALIAS libtempo-distance)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a test library
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_library(libtempo-distance-test INTERFACE)
target_sources(libtempo-distance-test INTERFACE
    test/multivariate_dep/m_adtw.cpp
    test/multivariate_dep/m_cdtw.cpp
    test/multivariate_dep/m_direct.cpp
    test/multivariate_dep/m_dtw.cpp
    test/multivariate_dep/m_erp.cpp
    test/multivariate_dep/m_lcss.cpp
    test/multivariate_dep/m_msm.cpp
    test/multivariate_dep/m_twe.cpp
    #test/multivariate_dep/m_wdtw.cpp
    test/univariate/adtw.cpp
    test/univariate/cdtw.cpp
    test/univariate/direct.cpp
    test/univariate/dtw.cpp
    test/univariate/erp.cpp
    test/univariate/lcss.cpp
    test/univariate/msm.cpp
    test/univariate/twe.cpp
    test/univariate/wdtw.cpp
)

add_library(libtempo::test::distance ALIAS libtempo-distance-test)
