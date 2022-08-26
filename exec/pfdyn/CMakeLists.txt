add_executable(pfdyn)
target_sources(pfdyn PRIVATE main.cpp pch.h)
target_link_libraries(pfdyn PUBLIC libtempo PRIVATE cxxopts)