cmake_minimum_required(VERSION 3.20)
project(CVSuite VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)

find_package(OpenCV REQUIRED)

# ─── System dependencies ──────────────────────────────────────────────────────

find_package(PkgConfig REQUIRED)
find_package(glfw3 REQUIRED)

# ─── ImGui — manual source integration via FetchContent ──────────────────────
# Check latest tag at: https://github.com/ocornut/imgui/releases

include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.6
)
FetchContent_MakeAvailable(imgui)

set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# ─── Application sources ──────────────────────────────────────────────────────

set(APP_SOURCES
    main.cpp
    app/App.cpp
    core/ImageLoader.cpp
    core/Utils.cpp
    modules/ThresholdModule.cpp
    modules/EdgeDetectionModule.cpp
)

add_executable(${PROJECT_NAME} ${APP_SOURCES} ${IMGUI_SOURCES})

# ─── macOS RPATH fix (Homebrew) ──────────────────────────────────────────────
if(APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "/opt/homebrew/opt/opencv/lib"
        INSTALL_RPATH "/opt/homebrew/opt/opencv/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# ─── Include paths ────────────────────────────────────────────────────────────

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}           # For #include "app/...", "core/...", etc.
    ${OpenCV_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${imgui_SOURCE_DIR}           # For #include "imgui.h"
    ${imgui_SOURCE_DIR}/backends  # For #include "backends/imgui_impl_*.h"
)

# ─── Compile definitions ──────────────────────────────────────────────────────

target_compile_definitions(${PROJECT_NAME} PRIVATE
    GL_SILENCE_DEPRECATION   # Suppress macOS OpenGL deprecation warnings
    GLFW_INCLUDE_NONE        # Prevent GLFW from pulling in <OpenGL/gl.h>;
                             # we manually include <OpenGL/gl3.h> for core profile
)

# ─── Link libraries ───────────────────────────────────────────────────────────

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    glfw

    "-framework OpenGL"
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
)

target_link_directories(${PROJECT_NAME} PRIVATE ${GLFW_LIBRARY_DIRS})

# ─── Build output ─────────────────────────────────────────────────────────────

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "GLFW   version : ${GLFW_VERSION}")
