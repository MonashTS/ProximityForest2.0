target_sources(libtempo
        PUBLIC
        reader.hpp
        ts/ts.hpp
        PRIVATE
        reader.cpp
        ts/ts.cpp
        )

add_subdirectory(csv)