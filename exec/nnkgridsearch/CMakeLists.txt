add_executable(nnkgridsearch)
target_sources(nnkgridsearch PRIVATE main.cpp pch.h cli.hpp cli.cpp)
target_link_libraries(nnkgridsearch PRIVATE libtempo)
target_precompile_headers(nnkgridsearch PRIVATE pch.h)
