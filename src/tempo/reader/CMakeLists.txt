target_sources(libtempo
        PUBLIC
        reader_result.hpp
        )

add_subdirectory(csv)


target_sources(libtempo-test
        PRIVATE
        #csv/csv.test.cpp
        )