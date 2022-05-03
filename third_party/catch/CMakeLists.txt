### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Setting up the test library catch2
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_library(Catch2 INTERFACE)

target_sources(Catch2
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/catch.hpp
)

set_target_properties(Catch2 PROPERTIES LANGUAGE CXX)
set_target_properties(Catch2 PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(Catch2 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
