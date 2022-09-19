add_executable(tschief)
target_sources(tschief PRIVATE main.cpp cmdline.cpp cmdline.hpp pfsplitters.hpp pfsplitters.cpp)
target_link_libraries(tschief PUBLIC libtempo tclap)