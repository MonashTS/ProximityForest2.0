add_executable(nnk)

target_sources(nnk
        PRIVATE
            main.cpp
        )


target_link_libraries(nnk PRIVATE libtempo armadillo jsoncpp)
target_precompile_headers(nnk PRIVATE pch.hpp)
