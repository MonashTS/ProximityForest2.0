target_sources(libtempo
        PUBLIC
        # --- --- --- Interfaces
        treedata.hpp
        treestate.hpp
        splitter_interface.hpp
        # --- --- --- Tree/Forest
        tree.hpp
        # --- --- --- Base splitter
        PRIVATE
        tree.cpp
        treestate.cpp
)

# Leaf splitters
add_subdirectory(sleaf)

# Node splitters
add_subdirectory(snode/meta)
add_subdirectory(snode/nn1splitter)