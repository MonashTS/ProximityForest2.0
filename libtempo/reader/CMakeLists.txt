### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create a libtempo::tseries component
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_library(libtempo-reader)
add_library(libtempo::reader ALIAS libtempo-reader)

target_include_directories(libtempo-reader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_sources(libtempo-reader
        PUBLIC
            ts/ts.hpp
            ts/ts.cpp
        PRIVATE
        readingtools.hpp
        )

target_link_libraries(libtempo-reader PUBLIC libtempo::tseries)


