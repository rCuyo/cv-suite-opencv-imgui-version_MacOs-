# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run

```bash
cmake -B build
cmake --build build
./build/bin/CVSuite
```

Requires macOS with Homebrew OpenCV at `/opt/homebrew/opt/opencv`. The CMakeLists.txt hardcodes this path (line 8). No automated test suite exists — validation is done by running the GUI application.

## Architecture

CVSuite is a desktop image processing tool built with OpenCV + ImGui + OpenGL 3.3. It provides a GUI for applying computer vision algorithms to loaded images.

**Entry point**: `main.cpp` creates an `App`, calls `init()` then `run()`.

**Three-layer structure**:
- `app/` — `App` class owns the GLFW window, ImGui context, and active modules. Its `renderUI()` draws a fixed 210px sidebar (navigation) and a scrollable content area (active module).
- `modules/` — Self-contained processing modules. Each module owns its parameters and renders its own ImGui controls + image output. Currently: `ThresholdModule` and `EdgeDetectionModule`.
- `core/` — `ImageLoader` wraps `cv::imread`; `Utils` handles `cv::Mat → OpenGL texture` conversion (BGR→RGB, 1-byte alignment, `GL_TEXTURE_2D` upload).

**Adding a new module**: create files in `modules/`, add it to `App`'s sidebar navigation and content switch, instantiate it in `App::init()`.

**Processing flow**: image load → OpenCV processing → `Utils::matToTexture()` → `ImGui::Image()` for display. Textures are re-uploaded on every parameter change.

### Module details

`ThresholdModule` — three use-case modes:
- **OCR**: Otsu + Adaptive (block size, C constant)
- **Medical Segmentation**: two-level intensity-band thresholding (lo/hi threshold pair)
- **Industrial Inspection**: blur + threshold for defect detection (sensitivity)

`EdgeDetectionModule` — two modes:
- **OCR/Document**: Canny edge detection (Gaussian kernel, low/high thresholds, Sobel aperture)
- **Segmentation**: contour extraction + visualization (min contour area)

## Key Dependencies

| Dependency | How it arrives |
|-----------|---------------|
| OpenCV 4.x | Homebrew (`/opt/homebrew/opt/opencv`) |
| ImGui v1.91.6 | CMake `FetchContent` (auto-downloaded) |
| GLFW3 | System via `PkgConfig` |
| OpenGL 3.3 core | macOS system frameworks |

macOS frameworks linked explicitly: `Cocoa`, `CoreVideo`, `IOKit`.

## CMake Notes

- C++17, `CMAKE_CXX_EXTENSIONS OFF`
- Binary output: `build/bin/CVSuite`
- OpenGL 3.3 core profile hint set in GLFW init code (required on macOS)
- `GL_SILENCE_DEPRECATION` and `GLFW_INCLUDE_NONE` defined globally
