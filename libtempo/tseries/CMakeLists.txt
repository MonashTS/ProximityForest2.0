### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::tseries component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_library(libtempo-tseries
        tseries.hpp
        dataset.hpp
        )

set_target_properties(libtempo-tseries PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(libtempo-tseries INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(libtempo-tseries PUBLIC armadillo)
target_link_libraries(libtempo-tseries PUBLIC nlohmann_json::nlohmann_json)

add_library(libtempo::tseries ALIAS libtempo-tseries)
