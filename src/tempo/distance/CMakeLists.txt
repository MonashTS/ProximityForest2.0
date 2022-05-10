target_sources(libtempo
        INTERFACE
        cost_function.hpp
        elastic/dtw.hpp
        elastic/adtw.hpp
        elastic/erp.hpp
        elastic/lcss.hpp
        elastic/msm.hpp
        elastic/twe.hpp
        elastic/wdtw.hpp
        lockstep/direct.hpp
        lockstep/minkowski.hpp
        lockstep/lorentzian.hpp
        )

if (BUILD_TESTING)
    target_sources(libtempo-test
            PRIVATE
            test/multivariate_dep/m_adtw.cpp
            test/multivariate_dep/m_direct.cpp
            test/multivariate_dep/m_dtw.cpp
            test/multivariate_dep/m_dtw_full.cpp
            test/multivariate_dep/m_erp.cpp
            test/multivariate_dep/m_lcss.cpp
            test/multivariate_dep/m_msm.cpp
            test/multivariate_dep/m_twe.cpp
            test/multivariate_dep/m_wdtw.cpp
            test/univariate/adtw.cpp
            test/univariate/direct.cpp
            test/univariate/dtw.cpp
            test/univariate/dtw_full.cpp
            test/univariate/erp.cpp
            test/univariate/lcss.cpp
            test/univariate/msm.cpp
            test/univariate/twe.cpp
            test/univariate/wdtw.cpp
            )
endif ()