cmake_minimum_required(VERSION 3.5.1)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(BuildFbGemm)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(streaming_inference_modules_nn_backend
        ${CMAKE_CURRENT_LIST_DIR}/Conv1dFbGemm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/LinearFbGemm.cpp
        ${CMAKE_CURRENT_LIST_DIR}/PackedGemmMatrixFP16.cpp
        )

set_target_properties(
        streaming_inference_modules_nn_backend
        PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 14
)

add_dependencies(streaming_inference_modules_nn_backend fbgemm)
add_dependencies(streaming_inference_modules_nn_backend cereal)
add_dependencies(streaming_inference_modules_nn_backend torch)

target_link_libraries(streaming_inference_modules_nn_backend
        PUBLIC
        streaming_inference_modules_nn
        ${fbgemm_LIBRARIES}
        ${TORCH_LIBRARIES}
        )

target_include_directories(streaming_inference_modules_nn_backend
        PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${fbgemm_INCLUDE_DIRS}
        ${cereal_INCLUDE_DIRS}
        ${wav2letter-inference_SOURCE_DIR}
        )

set(BACKEND_FOUND true)
