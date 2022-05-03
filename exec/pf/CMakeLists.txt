add_executable(pf)

target_sources(pf
        PRIVATE
            main.cpp
        )


target_link_libraries(pf libtempo armadillo nlohmann_json::nlohmann_json)
target_precompile_headers(pf PRIVATE pch.hpp)
