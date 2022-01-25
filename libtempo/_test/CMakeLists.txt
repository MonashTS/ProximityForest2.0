### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Setting up the test library catch2
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_library(Catch)

target_sources(Catch
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/catch/catch.hpp
            tests.cpp
)

target_include_directories(Catch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/catch)
target_include_directories(Catch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
