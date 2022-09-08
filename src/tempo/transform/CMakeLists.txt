add_subdirectory(core)

target_sources(libtempo
        # Implementation
        PUBLIC
        univariate.hpp
        tseries.univariate.hpp
        PRIVATE
        univariate.cpp
        univariate.private.hpp
        tseries.univariate.cpp
        )
