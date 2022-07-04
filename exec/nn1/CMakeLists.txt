add_executable(nn1)
target_sources(nn1 PRIVATE main.cpp pch.h cli.hpp cli.cpp)
target_link_libraries(nn1 PRIVATE libtempo)
target_precompile_headers(nn1 PRIVATE pch.h)
