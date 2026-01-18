cmake_minimum_required(VERSION 3.16)
project(vision)

find_package(OpenCV 4.7.0 REQUIRED)

set(CMAKE_SKIP_BUILD_RPATH           FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH    TRUE) 
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

fins_add_node(${PROJECT_NAME} src/nodes.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})