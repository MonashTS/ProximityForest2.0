cmake_minimum_required(VERSION 3.22)

message("Entering ${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(testlibs)
target_sources(testlibs PRIVATE testlibs.cpp)

#message(STATUS "MLPACK_INCLUDE_DIRS: ${MLPACK_INCLUDE_DIRS}")
#message(STATUS "MLPACK_LIBRARIES: ${MLPACK_LIBRARIES}")
target_include_directories(testlibs PRIVATE ${MLPACK_INCLUDE_DIRS})
target_link_libraries(testlibs PRIVATE ${MLPACK_LIBRARIES})


target_link_libraries(testlibs PRIVATE jsoncpp_lib jsoncpp_object JsonCpp::JsonCpp)


#find_library(OpenBLAS_LIBRARY OpenBLAS)
#target_include_directories(testlibs PRIVATE ${OpenBLAS_LIBRARY})
#target_link_libraries(testlibs PRIVATE ${OpenBLAS_LIBRARY})

#target_link_libraries(testlibs PRIVATE ${ARMADILLO_LIBRARIES})



