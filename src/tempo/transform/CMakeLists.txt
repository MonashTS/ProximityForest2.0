add_subdirectory(core)

target_sources(libtempo
        # Implementation
        PUBLIC
        univariate.hpp
        PRIVATE
        univariate.cpp
        univariate.private.hpp
        )
