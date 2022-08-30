add_executable(pf2018)
target_sources(pf2018 PRIVATE main.cpp cmdline.cpp cmdline.hpp)
target_link_libraries(pf2018 PUBLIC libtempo tclap)