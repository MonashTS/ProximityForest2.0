add_executable(pf2)
target_sources(pf2 PRIVATE main.cpp cmdline.cpp cmdline.hpp)
target_link_libraries(pf2 PUBLIC libtempo tclap)