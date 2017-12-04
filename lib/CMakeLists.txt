cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

find_package(Caffe REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories( ${OPENCV_INCLUDE_DIR} )

add_definitions(${Caffe_DEFINITIONS})

add_library(SegClassifier SHARED classifier.cpp)
target_link_libraries(SegClassifier ${Caffe_LIBRARIES})
target_link_libraries(SegClassifier ${OpenCV_LIBS})

install(TARGETS SegClassifier LIBRARY DESTINATION install/lib)
