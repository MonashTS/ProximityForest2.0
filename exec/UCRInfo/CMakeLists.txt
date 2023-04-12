add_executable(UCRInfo)
target_sources(UCRInfo PRIVATE main.cpp pch.h cli.hpp cli.cpp)
target_link_libraries(UCRInfo PRIVATE libtempo)
target_precompile_headers(UCRInfo PRIVATE pch.h)
