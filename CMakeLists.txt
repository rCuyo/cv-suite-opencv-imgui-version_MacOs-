cmake_minimum_required(VERSION 3.21)   # 3.21+ para TARGET_RUNTIME_DLLS
project(CVSuite VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# macOS sin vcpkg: apunta a Homebrew para que find_package encuentre OpenCV y GLFW
if(APPLE AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    list(PREPEND CMAKE_PREFIX_PATH /opt/homebrew)
endif()

find_package(OpenCV REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

if(WIN32)
    find_package(glad CONFIG REQUIRED)
endif()

# ─── ImGui — integración de fuentes via FetchContent ─────────────────────────
# Revisa el último tag en: https://github.com/ocornut/imgui/releases

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

# ─── tinyfiledialogs — diálogos de archivo nativos (cross-platform) ──────────

FetchContent_Declare(
    tinyfiledialogs
    GIT_REPOSITORY https://github.com/native-toolkit/libtinyfiledialogs.git
    GIT_SHALLOW    TRUE
    SOURCE_DIR     ${CMAKE_BINARY_DIR}/_deps/tinyfiledialogs-src
)
FetchContent_GetProperties(tinyfiledialogs)
if(NOT tinyfiledialogs_POPULATED)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(tinyfiledialogs)
endif()

# ─── Fuentes de la aplicación ─────────────────────────────────────────────────

set(APP_SOURCES
    main.cpp
    app/App.cpp
    core/ImageLoader.cpp
    core/Utils.cpp
    core/HistogramUtils.cpp
    core/FileDialog.cpp
    modules/ThresholdModule.cpp
    modules/EdgeDetectionModule.cpp
    ${tinyfiledialogs_SOURCE_DIR}/tinyfiledialogs.c
)

add_executable(${PROJECT_NAME} ${APP_SOURCES} ${IMGUI_SOURCES})

# ─── macOS: RPATH para encontrar las dylibs de Homebrew en runtime ────────────
if(APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "/opt/homebrew/opt/opencv/lib"
        INSTALL_RPATH "/opt/homebrew/opt/opencv/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# ─── Rutas de includes ────────────────────────────────────────────────────────

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}                    # Para #include "app/...", "core/...", etc.
    ${OpenCV_INCLUDE_DIRS}
    ${imgui_SOURCE_DIR}                    # Para #include "imgui.h"
    ${imgui_SOURCE_DIR}/backends           # Para #include "backends/imgui_impl_*.h"
    ${tinyfiledialogs_SOURCE_DIR}          # Para #include "tinyfiledialogs.h"
)

# ─── Definiciones de compilación ──────────────────────────────────────────────

target_compile_definitions(${PROJECT_NAME} PRIVATE
    GLFW_INCLUDE_NONE
    $<$<PLATFORM_ID:Darwin>:GL_SILENCE_DEPRECATION>
)

# UTF-8 en MSVC: necesario para caracteres españoles acentuados en literales de cadena
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
endif()

# ─── Enlace de librerías ──────────────────────────────────────────────────────

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    glfw
    $<$<PLATFORM_ID:Windows>:glad::glad>
    $<$<PLATFORM_ID:Windows>:opengl32>
    $<$<PLATFORM_ID:Windows>:comdlg32>
    $<$<PLATFORM_ID:Windows>:ole32>
    $<$<PLATFORM_ID:Darwin>:-framework OpenGL>
    $<$<PLATFORM_ID:Darwin>:-framework Cocoa>
    $<$<PLATFORM_ID:Darwin>:-framework IOKit>
    $<$<PLATFORM_ID:Darwin>:-framework CoreVideo>
)

# ─── Directorio de salida ─────────────────────────────────────────────────────

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ─── Windows: copiar DLLs al lado del ejecutable después del build ────────────
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND_EXPAND_LISTS
    )
endif()

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "GLFW   version : ${GLFW_VERSION}")
