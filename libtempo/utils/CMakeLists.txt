### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::utils component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_library(libtempo-utils SHARED
        capsule.hpp
        jsonvalue.hpp
        uncopyable.hpp
        utils.hpp
        utils.cpp
        stats.hpp
        partasks.hpp
        progressmonitor.hpp
        timing.hpp
        )

target_include_directories(libtempo-tseries INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_library(libtempo::utils ALIAS libtempo-utils)
