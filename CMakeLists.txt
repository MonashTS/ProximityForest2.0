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

# Armadillo
# --- Declare as "SYSTEM" to avoid its warning
# --- in RELEASE, do not check bounds (ARMA_NO_DEBUG)
add_subdirectory(third_party/armadillo-11.0.1)
get_target_property(_inc armadillo INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(armadillo SYSTEM INTERFACE ${_inc})
add_compile_definitions(armadillo INTERFACE $<$<CONFIG:RELEASE>:ARMA_NO_DEBUG>)

# Json
# Turn off library's tests
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(third_party/json-3.10.5)

add_subdirectory(libtempo)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Load executable
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#add_subdirectory(exec/nn1)
