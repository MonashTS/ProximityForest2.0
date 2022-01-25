### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::tseries component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

find_package(Eigen3 3.4 REQUIRED NO_MODULE PATHS ${CMAKE_CURRENT_LIST_FILE}/third_party/Eigen NO_DEFAULT_PATHS)

add_library(libtempo-tseries INTERFACE
        tseries.hpp
        )

target_include_directories(libtempo-tseries INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(libtempo-tseries INTERFACE Eigen3::Eigen)
add_library(libtempo::tseries ALIAS libtempo-tseries)
