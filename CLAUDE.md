# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run

**macOS**
```bash
cmake -B build
cmake --build build
./build/bin/CVSuite
```
Requires Homebrew at `/opt/homebrew` (OpenCV and GLFW resolved via `CMAKE_PREFIX_PATH`). ImGui and tinyfiledialogs are fetched automatically by CMake on first build.

**Windows**
```bat
bootstrap.bat          # configures vcpkg + CMake + builds
build\bin\Release\CVSuite.exe
```
Or manually: `cmake -B build -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake && cmake --build build --config Release`

No automated test suite — validation is done by running the GUI application.

## Architecture

CVSuite is a desktop image processing tool built with OpenCV + ImGui + OpenGL 3.3. It provides a GUI for applying computer vision algorithms to loaded images.

**Entry point**: `main.cpp` creates an `App`, calls `init()` then `run()`.

**Three-layer structure**:
- `app/` — `App` class owns the GLFW window, ImGui context, and active modules. Its `renderUI()` draws a fixed 210px sidebar (navigation) and a scrollable content area (active module).
- `modules/` — Self-contained processing modules. Each module owns its parameters and renders its own ImGui controls, image output, and histograms.
- `core/` — Shared utilities: `ImageLoader` wraps `cv::imread`; `Utils` handles `cv::Mat → OpenGL texture` conversion (BGR→RGB, 1-byte alignment, `GL_TEXTURE_2D` upload); `HistogramUtils` computes normalized 256-bin histograms and per-image stats (mean, stddev, white/black pixel %); `FileDialog` wraps `tinyfiledialogs` for native open/save pickers; `gl.h` abstracts the OpenGL header (macOS uses `<OpenGL/gl3.h>` directly; Windows uses GLAD — always include `core/gl.h` before GLFW).

**UI language**: All user-facing strings are in Spanish. New controls must follow suit; accented characters (á é í ó ú ñ ü ¡ ¿) are covered by the Latin-1 font range already loaded in `App::init()`.

**Adding a new module**: create files in `modules/`, implement `renderUI()` and `loadAndProcess(const std::string& path)`, add it to `App`'s `ActiveModule` enum, sidebar nav (`renderSidebar()`), and content switch (`renderContent()`), then instantiate it in `App::init()`. The sidebar has a disabled "ML Comparativo" placeholder for the next planned module. Wrap the entire `renderUI()` body in `ImGui::PushID("ModuleName")` / `ImGui::PopID()` to avoid widget ID collisions between modules.

**Processing flow**: image load → `loadAndProcess()` → OpenCV processing → `Utils::matToTexture()` / `Utils::updateTexture()` → `ImGui::Image()` for display. Textures are re-uploaded on every parameter change via the `m_needsProcess` flag (set in ImGui callbacks, consumed in `renderUI()` to call `reprocess()`).

### Texture ownership

`Texture` (defined in `core/Utils.h`) is a plain struct holding a `GLuint id` + dimensions. Modules own `Texture` members and are responsible for calling `Utils::deleteTexture()` in their destructors. `Utils::updateTexture()` deletes the old GPU texture and uploads a fresh one — call it instead of `matToTexture()` when the slot already exists.

### Raw-image pattern

Modules store the originally-loaded image in a `m_rawBGR` member that is **never modified**. `reprocess()` derives all intermediate and output images from `m_rawBGR` each time, so any parameter toggle always starts from the unmodified source.

### Module details

`ThresholdModule` — three use-case modes:
- **OCR**: Otsu + Adaptive (block size, C constant)
- **Medical Segmentation**: two-level intensity-band thresholding (lo/hi threshold pair)
- **Industrial Inspection**: blur + threshold for defect detection (sensitivity)

Each mode renders two result images and their histograms side-by-side with the original.

`EdgeDetectionModule` — two modes:
- **OCR/Document**: Canny edge detection (Gaussian kernel, low/high thresholds, Sobel aperture)
- **Segmentation**: contour extraction + visualization (min contour area)

`DocumentCleanupModule` — document scanning / cleanup pipeline with toggleable stages:
1. **Perspective correction** — detects the largest 4-sided contour (downscaled to 800 px for speed), warps to a top-down view using `cv::getPerspectiveTransform`/`cv::warpPerspective`; falls back to full frame if no page boundary is found.
2. **Shadow reduction** — CLAHE (clip limit, 8×8 tile grid) before blur to preserve local detail.
3. **Noise reduction** — Gaussian blur (odd kernel, 1–15).
4. **Sharpening** — unsharp mask (`addWeighted`) with adjustable strength.
5. **Binary output** — adaptive Gaussian threshold (block size, C constant); optional morphological open to remove noise dots.

Displays original vs. cleaned result side by side with intensity histograms and an export button.

## Key Dependencies

| Dependency | How it arrives |
|-----------|---------------|
| OpenCV 4.x | Homebrew (macOS) / vcpkg `opencv4` (Windows) |
| ImGui v1.91.6 | CMake `FetchContent` (auto-downloaded) |
| tinyfiledialogs | CMake `FetchContent` (auto-downloaded) |
| GLFW3 | Homebrew (macOS) / vcpkg `glfw3` (Windows) |
| GLAD | vcpkg `glad` (Windows only) |
| OpenGL 3.3 core | macOS system frameworks / Windows system |

macOS frameworks linked explicitly: `OpenGL`, `Cocoa`, `CoreVideo`, `IOKit`.

## Windows support

The project builds on Windows via vcpkg (`vcpkg.json` at root declares `opencv4`, `glfw3`, `glad`). On Windows: GLAD is required (linked via `glad::glad`), `comdlg32`/`ole32` are linked for tinyfiledialogs, and MSVC gets `/utf-8` for Spanish string literals. `bootstrap.bat` automates the full vcpkg + CMake + build sequence.

## CMake Notes

- C++17, `CMAKE_CXX_EXTENSIONS OFF`
- Binary output: `build/bin/CVSuite` (macOS) / `build/bin/Release/CVSuite.exe` (Windows)
- OpenGL 3.3 core profile hint set in GLFW init (`GLFW_OPENGL_FORWARD_COMPAT` required on macOS)
- `GL_SILENCE_DEPRECATION` and `GLFW_INCLUDE_NONE` defined globally
- macOS RPATH set to `/opt/homebrew/opt/opencv/lib` so the binary finds OpenCV dylibs at runtime without `install_name_tool`
