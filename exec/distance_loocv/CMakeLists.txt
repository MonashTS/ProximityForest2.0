add_executable(distance_loocv)
target_sources(distance_loocv PRIVATE main.cpp)
target_link_libraries(distance_loocv PRIVATE libtempo)
