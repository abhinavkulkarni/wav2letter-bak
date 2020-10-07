cmake_minimum_required(VERSION 3.5.1)

add_library(streaming_inference_modules INTERFACE)

target_sources(streaming_inference_modules
        INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/InferenceModule.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ModuleParameter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ModuleProcessingState.cpp
        )

add_dependencies(streaming_inference_modules cereal)
add_dependencies(streaming_inference_modules torch)

target_link_libraries(
        streaming_inference_modules
        INTERFACE
        ${cereal_LIBRARIES}
        ${TORCH_LIBRARIES}
        streaming_inference_common
)

target_include_directories(
        streaming_inference_modules
        INTERFACE
        ${cereal_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        # going up one directory to use #include "inference/*/*.h"
        ${wav2letter-inference_SOURCE_DIR}
)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/nn)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/feature)