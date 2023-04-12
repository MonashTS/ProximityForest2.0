target_sources(libtempo
        # Implementation
        PUBLIC
        univariate.derivative.hpp
        univariate.noise.hpp
        univariate.normalization.hpp
        )

### Testing
if (BUILD_TESTING)
    target_sources(libtempo-test
            # --- --- ---
            PRIVATE
            univariate.derivative.test.cpp
            )
endif ()
