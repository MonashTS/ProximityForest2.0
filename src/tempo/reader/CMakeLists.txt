target_sources(libtempo
        PUBLIC
        ts/ts.hpp
        #reader.hpp
        new_reader.hpp
        csv/csv.hpp
        PRIVATE
        ts/ts.cpp
        csv/csv.cpp
        #reader.cpp
        )


target_sources(libtempo-test
        PRIVATE
        csv/csv.test.cpp
        )