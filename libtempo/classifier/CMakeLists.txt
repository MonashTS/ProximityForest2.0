### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Libtempo Classifier
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

add_library(libtempo-classifier INTERFACE)
add_library(libtempo::classifier ALIAS libtempo-classifier)

target_include_directories(libtempo-classifier INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_sources(libtempo-classifier
        INTERFACE
        splitting_forest/ipf.hpp
        splitting_forest/proximity_forest/pf2018.hpp
        splitting_forest/pftree.hpp
        splitting_forest/splitters.hpp
        splitting_forest/proximity_forest/distance_splitters.hpp
        )

target_link_libraries(libtempo-classifier INTERFACE libtempo::tseries)


