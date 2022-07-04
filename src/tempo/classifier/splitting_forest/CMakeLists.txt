### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Libtempo Classifier
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

target_sources(libtempo
        PRIVATE
            proximity_forest/pf2018.cpp
        PUBLIC
            ipf.hpp
            pftree.hpp
            splitters.hpp
            proximity_forest/distance_splitters.hpp
            proximity_forest/pf2018.hpp
            utils.hpp
        )