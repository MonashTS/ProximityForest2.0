add_executable(pf2f)
target_sources(pf2f PRIVATE main.cpp cmdline.cpp cmdline.hpp)
target_link_libraries(pf2f PUBLIC libtempo tclap fftw3)