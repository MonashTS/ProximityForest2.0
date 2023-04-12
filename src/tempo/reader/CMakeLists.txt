target_sources(libtempo
        PUBLIC
        reader.hpp
        ts/ts.hpp
        dts.reader.hpp
        PRIVATE
        reader.cpp
        ts/ts.cpp
        dts.reader.cpp
        )

add_subdirectory(csv)