### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::distance component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_library(libtempo-transform INTERFACE
        derivative.hpp
        )

target_include_directories(libtempo-transform INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_library(libtempo::transform ALIAS libtempo-transform)