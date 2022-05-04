add_executable(pf)

target_sources(pf
        PRIVATE
            main.cpp
        )


target_link_libraries(pf PRIVATE libtempo armadillo jsoncpp)
target_precompile_headers(pf PRIVATE pch.hpp)
