add_subdirectory(core)

target_sources(libtempo
        # Implementation
        PUBLIC
        univariate.hpp
        tseries.univariate.cpp
        PRIVATE
        univariate.cpp
        univariate.private.hpp
        )
