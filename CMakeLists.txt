cmake_minimum_required(VERSION 3.21)

project(tempo
        VERSION 0.0.1
        DESCRIPTION "Time Series classification library"
        LANGUAGES CXX)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Global flags
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-W -Wall)
    add_compile_options(-fverbose-asm -save-temps)
    add_compile_options(-fconcepts-diagnostics-depth=2)
    add_compile_definitions(INTERFACE $<$<CONFIG:DEBUG>:_GLIBCXX_ASSERTIONS>)
endif()



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Load subprojects
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_subdirectory(third_party/catch)
add_subdirectory(libtempo)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Load executable
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#add_subdirectory(exec/nn1)
