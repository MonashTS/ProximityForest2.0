add_executable(loocv)
target_sources(loocv PRIVATE main.cpp)
target_link_libraries(loocv PRIVATE libtempo)
