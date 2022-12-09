add_executable(mknoise)
target_sources(mknoise PRIVATE main.cpp)
target_link_libraries(mknoise PRIVATE libtempo)
