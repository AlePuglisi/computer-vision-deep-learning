cmake_minimum_required(VERSION 3.5)

project( display_image )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( display_image main.cpp )

target_link_libraries( display_image
         ${OpenCV_LIBS} 
)