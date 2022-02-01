cmake_minimum_required(VERSION 3.19)

project(tempo
        VERSION 0.0.1
        DESCRIPTION "Time Series classification library"
        LANGUAGES CXX)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Global flags
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_definitions("-fverbose-asm")

if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-W -Wall)
endif()



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Load subprojects
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_subdirectory(third_party/catch)
add_subdirectory(libtempo)
