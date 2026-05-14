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

**Entry point**: `main.cpp` creates an `App` at 1280×800 ("CV Suite v1.2"), calls `init()` then `run()`. GLSL version hint: `#version 150`.

**Three-layer structure**:
- `app/` — `App` class owns the GLFW window, ImGui context, and active modules. Its `renderUI()` draws a fixed 210px sidebar (navigation) and a scrollable content area (active module).
- `modules/` — Self-contained processing modules. Each module owns its parameters and renders its own ImGui controls, image output, and histograms.
- `core/` — Shared utilities:
  - `ImageLoader` wraps `cv::imread`
  - `Utils` handles `cv::Mat → OpenGL texture` conversion (BGR→RGB, 1-byte alignment, `GL_TEXTURE_2D` upload)
  - `HistogramUtils` computes normalized 256-bin histograms and per-image stats (mean, stddev, white/black pixel %)
  - `HistogramRenderer` renders a histogram `std::array<float,256>` to a `cv::Mat` image using OpenCV drawing primitives; also provides `renderAndSave()` for direct PNG export without screenshots
  - `ExportUtils::nextExportPath(dir, prefix, ext)` generates the next available numbered filename (e.g. `ocr_resultado_3.png`) to avoid overwriting previous exports
  - `FileDialog` wraps `tinyfiledialogs` for native open/save pickers
  - `gl.h` abstracts the OpenGL header (macOS uses `<OpenGL/gl3.h>` directly; Windows uses GLAD — always include `core/gl.h` before GLFW)

**UI language**: All user-facing strings are in Spanish. New controls must follow suit; accented characters (á é í ó ú ñ ü ¡ ¿) are covered by the Latin-1 font range (0x0020–0x00FF) already loaded in `App::init()`. Platform fonts: Segoe UI 15px on Windows, Helvetica 15px on macOS.

**Adding a new module**: create files in `modules/`, implement `renderUI()` and `loadAndProcess(const std::string& path)`, add it to `App`'s `ActiveModule` enum, sidebar nav (`renderSidebar()`), and content switch (`renderContent()`), then instantiate it in `App::init()`. The sidebar is divided into two color-coded groups (MÓDULO 1: Segmentación y Bordes; MÓDULO 2: Transf. y Mejora de Imagen) — add the nav button in the appropriate group. Wrap the entire `renderUI()` body in `ImGui::PushID("ModuleName")` / `ImGui::PopID()` to avoid widget ID collisions between modules.

**Processing flow**: image load → `loadAndProcess()` → OpenCV processing → `Utils::matToTexture()` / `Utils::updateTexture()` → `ImGui::Image()` for display. Textures are re-uploaded on every parameter change via the `m_needsProcess` flag (set in ImGui callbacks, consumed in `renderUI()` to call `reprocess()`).

### Texture ownership

`Texture` (defined in `core/Utils.h`) is a plain struct holding a `GLuint id` + dimensions. Modules own `Texture` members and are responsible for calling `Utils::deleteTexture()` in their destructors. `Utils::updateTexture()` deletes the old GPU texture and uploads a fresh one — call it instead of `matToTexture()` when the slot already exists.

### Raw-image pattern

Most modules store the originally-loaded image in a `m_rawBGR` (or `m_original`) member that is **never modified**. `reprocess()` derives all intermediate and output images from it each time, so any parameter toggle always starts from the unmodified source.

`DocumentCleanupModule` is the exception: it keeps both `m_rawBGR` (disk image, truly untouched) and `m_original` (perspective-corrected view, shown as "Original" in the UI). When perspective correction is toggled, `reprocess()` re-derives `m_original` from `m_rawBGR`.

### Export feedback pattern

Modules that export files use `m_exportMsg` (string) and `m_exportMsgTime` (double, `glfwGetTime()` timestamp) to display a timed confirmation message in the UI. Check the message against the current time in `renderUI()` and clear it after a few seconds.

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

`TransformModule` — three geometric transform modes applied via OpenCV affine/resize ops:
- **Translation**: `warpAffine` shift along X/Y; supports interactive drag on the result image.
- **Rotation**: `getRotationMatrix2D` around image center, continuous angle slider.
- **Scaling**: `cv::resize` with independent X/Y scale factors; uniform-scale and lock-aspect-ratio toggles.

`HistogramEqualizationModule` — converts the loaded image to grayscale, then applies `cv::equalizeHist`. Displays grayscale vs. equalized images with before/after histograms and stats.

`NoiseReductionModule` — three filter modes (enum `NoiseFilterMode`):
- **Gaussian**: `cv::GaussianBlur` — odd kernel size, optional sigma (0 = auto).
- **Median**: `cv::medianBlur` — robust to salt-and-pepper noise, odd kernel size.
- **Bilateral**: `cv::bilateralFilter` — edge-preserving; diameter, sigmaColor, sigmaSpace.

Shows processing time (`m_lastProcessMs`) next to each result.

`MorphologyModule` — four morphological operations (enum `MorphOp`): Erosion, Dilation, Opening, Closing; applied with a configurable kernel shape (enum `KernelShape`: Rectangle, Ellipse, Cross), kernel size (odd, 1–21), and iteration count (1–10). Shows processing time alongside the result.

`HOGPedestrianModule` — dos modos de detección de peatones con `cv::HOGDescriptor` + SVM preentrenado (`getDefaultPeopleDetector()`):

**Image Mode** (modo interactivo):
- Carga una imagen via `FileDialog::openImage()`; detección re-ejecutada en la UI thread con `m_imageNeedsDetect`
- Muestra original + resultado con bounding boxes verdes; exporta con `ExportUtils::nextExportPath()`

**Video Mode** (procesamiento offline):
- Carga metadatos del video sin reproducirlo (`cv::VideoCapture::get()` solamente)
- "Start Processing" lanza `std::thread` (worker thread) que procesa frame a frame con su propio `cv::HOGDescriptor` local (no comparte `m_hog` para evitar data races)
- La UI thread solo muestra barra de progreso y estadísticas leyendo `std::atomic<int> m_processedFrames` / `m_detectedTotal`
- "Cancel Processing" pone `m_cancelRequested = true`; el worker limpia el archivo parcial al cancelar
- `pollWorker()` se llama cada render frame: cuando `m_isProcessing` pasa a false, hace `join()` y muestra resultado
- Salida: AVI con codec MJPG (`VideoWriter::fourcc('M','J','P','G')`) → compatible sin ffmpeg
- Carpeta de salida seleccionable via `FileDialog::openFolder()` (por defecto `exports/`)
- Nombre del archivo: `{stem}_hog_processed.avi`

Parámetros compartidos por ambos modos: Hit Threshold, Win Stride X/Y, Scale, Group Threshold.

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
