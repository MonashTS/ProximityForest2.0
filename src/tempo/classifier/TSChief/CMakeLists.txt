target_sources(libtempo
        PUBLIC
        # --- --- --- Interfaces
        treedata.hpp
        treestate.hpp
        splitter_interface.hpp
        # --- --- --- Tree/Forest
        tree.hpp
        forest.hpp
        # --- --- --- Base splitter
        PRIVATE
        treestate.cpp
        tree.cpp
        forest.cpp
)

# Leaf splitters
add_subdirectory(sleaf)

# Node splitters
add_subdirectory(snode/meta)
add_subdirectory(snode/nn1splitter)