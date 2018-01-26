cmake_minimum_required(VERSION 3.6)
project(charuco-sample-project)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
#find_package(FREEGLUT REQUIRED)
find_package(GLUT REQUIRED)
find_package (Threads)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp Charuco.cpp Charuco.h)

include_directories(${GLUT_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})

file(COPY resources DESTINATION ./)

add_executable(charuco-sample-project ${SOURCE_FILES})
#configure_file(resources/fjords.jpg fjords.jpg COPYONLY)



target_link_libraries(charuco-sample-project ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
