add_executable(nnkgridsearch)
target_sources(nnkgridsearch PRIVATE main.cpp cli.hpp cli.cpp)
target_link_libraries(nnkgridsearch PRIVATE libtempo)
