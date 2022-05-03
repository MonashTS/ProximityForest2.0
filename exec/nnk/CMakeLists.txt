add_executable(nnk)

target_sources(nnk
        PRIVATE
            main.cpp
        )


target_link_libraries(nnk libtempo armadillo nlohmann_json::nlohmann_json)
target_precompile_headers(nnk PRIVATE pch.hpp)
