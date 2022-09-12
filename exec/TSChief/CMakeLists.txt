add_executable(tschief)
target_sources(tschief PRIVATE main.cpp cmdline.cpp cmdline.hpp pf2018.hpp pf2018.cpp)
target_link_libraries(tschief PUBLIC libtempo tclap)