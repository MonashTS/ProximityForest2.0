add_executable(tschief)
target_sources(tschief PRIVATE main.cpp cmdline.cpp cmdline.hpp)
target_link_libraries(tschief PUBLIC libtempo tclap)