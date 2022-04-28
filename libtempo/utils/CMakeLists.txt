### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::utils component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_library(libtempo-utils SHARED
        jsonvalue.hpp
        uncopyable.hpp
        stats.hpp
        utils.hpp
        utils.cpp
        )

target_include_directories(libtempo-tseries INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_library(libtempo::utils ALIAS libtempo-utils)
