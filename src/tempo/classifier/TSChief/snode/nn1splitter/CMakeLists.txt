target_sources(libtempo
        # --- --- ---
        # --- --- ---
        # --- --- ---
        PUBLIC
        # General interface
        nn1splitter.hpp
        nn1dist.hpp
        nn1dist_base.hpp
        # Specific implementation
        nn1_directa.hpp
        nn1_dtwfull.hpp

        # --- --- ---
        # --- --- ---
        # --- --- ---
        PRIVATE
        nn1splitter.private.hpp
        nn1splitter.cpp
)