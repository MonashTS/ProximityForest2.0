target_sources(libtempo
        PUBLIC
        univariate/csv.hpp
        PRIVATE
        univariate/csv.template.cpp
        univariate/csv.cpp
        )


### Testing
if (BUILD_TESTING)
    target_sources(libtempo-test
        # --- --- ---
        PRIVATE
        univariate/csv.test.cpp
    )
endif ()
