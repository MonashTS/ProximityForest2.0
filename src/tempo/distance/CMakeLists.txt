add_subdirectory(core)

target_sources(libtempo
        # Implementation
        PUBLIC
        utils.hpp
        cost_functions.hpp
        univariate.hpp
        tseries.univariate.hpp
        PRIVATE
        univariate.private.hpp
        univariate.cpp
        )
