### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Libtempo Classifier
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_library(libtempo-classifier INTERFACE)
add_library(libtempo::classifier ALIAS libtempo-classifier)

target_include_directories(libtempo-classifier INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_sources(libtempo-classifier
        INTERFACE
        proximity_forest/pftree.hpp
        proximity_forest/ipf.hpp
        proximity_forest/splitters.hpp
        proximity_forest/splitters/distance_splitters.hpp
        )

target_link_libraries(libtempo-classifier INTERFACE libtempo::tseries)


