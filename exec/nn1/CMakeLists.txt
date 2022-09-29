add_executable(nn1)
target_sources(nn1 PRIVATE main.cpp cli.hpp cli.cpp)
target_link_libraries(nn1 PRIVATE libtempo)
