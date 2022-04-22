### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::tseries component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_library(libtempo-tseries INTERFACE
        tseries.hpp
        dataset.hpp
        )

target_include_directories(libtempo-tseries INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(libtempo-tseries INTERFACE armadillo)
add_library(libtempo::tseries ALIAS libtempo-tseries)
