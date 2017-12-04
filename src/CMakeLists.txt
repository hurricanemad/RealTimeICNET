cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

ADD_DEFINITIONS(-DREENTRANT)
find_package(Caffe REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories( ${OPENCV_INCLUDE_DIR} )
include_directories(../lib)
link_directories(lib)

add_executable(videocapture main.cpp prefix.hpp)

target_link_libraries(videocapture ${Caffe_LIBRARIES})
target_link_libraries(videocapture ${OpenCV_LIBS})
target_link_libraries(videocapture SegClassifier)

install(TARGETS videocapture RUNTIME DESTINATION install)
