### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::utils component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

target_sources(libtempo
        PRIVATE
            utils.cpp
        PUBLIC
            utils/uncopyable.hpp
            utils/stats.hpp
            concepts.hpp
            utils.hpp
        )

target_precompile_headers(libtempo PRIVATE utils.hpp)