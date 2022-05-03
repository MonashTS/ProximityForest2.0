### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Libtempo Classifier
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

target_sources(libtempo
        INTERFACE
            splitting_forest/ipf.hpp
            splitting_forest/proximity_forest/pf2018.hpp
            splitting_forest/pftree.hpp
            splitting_forest/splitters.hpp
            splitting_forest/proximity_forest/distance_splitters.hpp
        )