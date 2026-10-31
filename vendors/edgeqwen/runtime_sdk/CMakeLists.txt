cmake_minimum_required(VERSION 3.18)

file(GLOB RUNTIME_SDK_CPP_SOURCES "src/*.cpp")

add_library(edgeqwen_runtime STATIC ${RUNTIME_SDK_CPP_SOURCES})

target_include_directories(
  edgeqwen_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(edgeqwen_runtime PRIVATE commonLibraryExt)
target_link_libraries(edgeqwen_runtime INTERFACE edgellmCore)

add_cross_build_link_options(edgeqwen_runtime)

add_executable(edgeqwen_runtime_demo demo/runtime_demo.cpp)
target_link_libraries(edgeqwen_runtime_demo PRIVATE edgeqwen_runtime edgellmCore
                                                     commonLibraryExt)
add_cross_build_link_options(edgeqwen_runtime_demo)

add_executable(edgeqwen_runtime_stability_check demo/stability_check.cpp)
target_link_libraries(edgeqwen_runtime_stability_check PRIVATE edgeqwen_runtime edgellmCore
                                                                commonLibraryExt)
add_cross_build_link_options(edgeqwen_runtime_stability_check)
add_executable(qwen_ask demo/qwen_ask.cpp)
target_link_libraries(qwen_ask PRIVATE edgeqwen_runtime edgellmCore commonLibraryExt)
add_cross_build_link_options(qwen_ask)
