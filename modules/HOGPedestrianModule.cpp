#include "HOGPedestrianModule.h"
#include "core/FileDialog.h"
#include "core/ExportUtils.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────

HOGPedestrianModule::HOGPedestrianModule()
{
    m_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
}

HOGPedestrianModule::~HOGPedestrianModule()
{
    if (m_workerThread.joinable()) {
        m_cancelRequested = true;
        m_workerThread.join();
    }
    Utils::deleteTexture(m_texImageOrig);
    Utils::deleteTexture(m_texImageResult);
}

// ─── Top-level UI ─────────────────────────────────────────────────────────────

void HOGPedestrianModule::renderUI()
{
    ImGui::PushID("HOGPedestrianModule");

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.75f, 1.0f), "HoG Pedestrian Detection");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Explicación académica ─────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.55f, 0.55f, 1.0f));
    ImGui::TextWrapped(
        "HoG (Histogram of Oriented Gradients) describe la forma humana "
        "utilizando distribuciones de gradientes. HoG extrae caracteristicas, "
        "SVM clasifica peatones y funciona bien sin GPU.");
    ImGui::Spacing();
    ImGui::TextWrapped(
        "HoG + SVM puede ejecutarse eficientemente sobre CPU sin necesidad de GPU dedicada. "
        "El procesamiento offline permite analizar videos completos sin afectar el "
        "rendimiento de la interfaz grafica.");
    ImGui::PopStyleColor();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ── Mode selector ─────────────────────────────────────────────────────────
    bool processingActive = m_isProcessing.load();

    ImGui::Text("Modo:");
    ImGui::SameLine();

    auto modeBtn = [&](const char* label, HOGInputMode mode) {
        bool sel = (m_mode == mode);
        if (sel)
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (processingActive) ImGui::BeginDisabled();
        if (ImGui::Button(label, ImVec2(100, 0))) m_mode = mode;
        if (processingActive) ImGui::EndDisabled();
        if (sel) ImGui::PopStyleColor();
        ImGui::SameLine();
    };

    modeBtn("Imagen", HOGInputMode::Image);
    modeBtn("Video",  HOGInputMode::Video);
    ImGui::NewLine();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (m_mode == HOGInputMode::Image)
        renderImageMode();
    else
        renderVideoMode();

    ImGui::PopID();
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMAGE MODE — procesamiento interactivo sobre imagen estática
// ═══════════════════════════════════════════════════════════════════════════════

void HOGPedestrianModule::renderImageMode()
{
    // ── Browse ────────────────────────────────────────────────────────────────
    if (ImGui::Button("Browse Image...", ImVec2(130, 0))) {
        std::string p = FileDialog::openImage();
        if (!p.empty()) loadImage(p);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_imagePath.empty() ? "No hay imagen cargada." : m_imagePath.c_str());
    ImGui::Spacing();

    // ── Parameters ────────────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();
    renderParameters();

    // ── Lazy detection ────────────────────────────────────────────────────────
    if (m_imageNeedsDetect && !m_imageBGR.empty()) {
        detectOnImage();
        m_imageNeedsDetect = false;
    }

    // ── Images ────────────────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();
    drawImageRow();

    // ── Stats ─────────────────────────────────────────────────────────────────
    if (!m_imageBGR.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Informacion Tecnica");
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
        ImGui::Text("Peatones detectados : %d", m_imageDetCount);
        ImGui::Text("Resolucion          : %d x %d px", m_imageBGR.cols, m_imageBGR.rows);
        if (m_imageDetectMs > 0.0f)
            ImGui::Text("Tiempo deteccion    : %.1f ms", m_imageDetectMs);
        ImGui::PopStyleColor();
    }

    // ── Export ────────────────────────────────────────────────────────────────
    if (m_texImageResult.valid()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Export Current Frame", ImVec2(160, 0))) {
            std::string path = ExportUtils::nextExportPath("exports", "hog_imagen", "png");
            bool ok = cv::imwrite(path, m_imageProcessed);
            m_imageExportMsg  = ok ? ("Guardado: " + path) : "Error: no se pudo guardar.";
            m_imageExportTime = ImGui::GetTime();
        }

        if (!m_imageExportMsg.empty() && (ImGui::GetTime() - m_imageExportTime) < 4.0) {
            ImGui::SameLine();
            bool isErr = (m_imageExportMsg.rfind("Error", 0) == 0);
            ImGui::TextColored(
                isErr ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f) : ImVec4(0.4f, 1.0f, 0.6f, 1.0f),
                "%s", m_imageExportMsg.c_str());
        }
    }
}

void HOGPedestrianModule::loadImage(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[HOGPedestrian] Failed to load image: %s\n", path.c_str());
        return;
    }
    m_imageBGR         = img;
    m_imagePath        = path;
    m_imageNeedsDetect = true;
}

void HOGPedestrianModule::detectOnImage()
{
    if (m_imageBGR.empty()) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<cv::Rect> dets;
    m_hog.detectMultiScale(
        m_imageBGR, dets,
        m_hitThreshold,
        cv::Size(std::max(1, m_winStrideX), std::max(1, m_winStrideY)),
        cv::Size(0, 0),
        m_scale,
        static_cast<double>(m_groupThreshold));

    auto t1 = std::chrono::high_resolution_clock::now();
    m_imageDetectMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    m_imageDetCount = static_cast<int>(dets.size());

    m_imageBGR.copyTo(m_imageProcessed);
    for (const auto& r : dets) {
        cv::rectangle(m_imageProcessed, r, cv::Scalar(0, 200, 0), 2);
        cv::rectangle(m_imageProcessed,
                      cv::Point(r.x + 4, r.y + 4),
                      cv::Point(r.x + r.width - 4, r.y + r.height - 4),
                      cv::Scalar(0, 230, 80), 1);
    }

    Utils::updateTexture(m_texImageOrig,   m_imageBGR);
    Utils::updateTexture(m_texImageResult, m_imageProcessed);
}

void HOGPedestrianModule::drawImageRow()
{
    if (!m_texImageOrig.valid()) {
        ImGui::TextDisabled("No hay imagen cargada. Click en Browse Image...");
        return;
    }

    const float avail   = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float imgW    = (avail - spacing) / 2.0f;
    const float aspect  = (m_texImageOrig.width > 0)
                          ? static_cast<float>(m_texImageOrig.height) /
                            static_cast<float>(m_texImageOrig.width)
                          : 1.0f;
    const float imgH    = imgW * aspect;

    ImGui::Text("Imagen Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("Deteccion HoG  [%d peatones]", m_imageDetCount);

    ImGui::Image((ImTextureID)(uintptr_t)m_texImageOrig.id,   ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texImageResult.id, ImVec2(imgW, imgH));
}

// ═══════════════════════════════════════════════════════════════════════════════
// VIDEO MODE — procesamiento offline frame a frame con thread en segundo plano
// ═══════════════════════════════════════════════════════════════════════════════

void HOGPedestrianModule::renderVideoMode()
{
    pollWorker();

    bool processingActive = m_isProcessing.load();

    // ── Browse video ──────────────────────────────────────────────────────────
    if (processingActive) ImGui::BeginDisabled();
    if (ImGui::Button("Browse Video...", ImVec2(130, 0))) {
        std::string p = FileDialog::openVideo();
        if (!p.empty()) {
            loadVideoMeta(p);
            m_showResult = false;
        }
    }
    if (processingActive) ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_hasVideoInput ? m_videoInputPath.c_str()
                                              : "No hay video cargado.");
    ImGui::Spacing();

    // ── Choose output folder ──────────────────────────────────────────────────
    if (processingActive) ImGui::BeginDisabled();
    if (ImGui::Button("Carpeta de salida...", ImVec2(160, 0))) {
        std::string folder = FileDialog::openFolder();
        if (!folder.empty()) m_videoOutputDir = folder;
    }
    if (processingActive) ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_videoOutputDir.empty()
                              ? "exports/ (por defecto)"
                              : m_videoOutputDir.c_str());
    ImGui::Spacing();

    // ── Video metadata ────────────────────────────────────────────────────────
    if (m_hasVideoInput) {
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Informacion del video");
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
        ImGui::Text("Resolucion  : %d x %d px", m_vidWidth, m_vidHeight);
        ImGui::Text("FPS         : %.2f", m_vidFps);
        int durMin = static_cast<int>(m_vidDurationSec) / 60;
        int durSec = static_cast<int>(m_vidDurationSec) % 60;
        ImGui::Text("Duracion    : %d:%02d", durMin, durSec);
        ImGui::Text("Frames      : %d", m_vidTotalFrames);
        if (m_vidFileSizeBytes > 0)
            ImGui::Text("Tamano      : %.1f MB",
                        static_cast<double>(m_vidFileSizeBytes) / (1024.0 * 1024.0));
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // ── Parameters (solo visibles cuando no se procesa) ───────────────────────
    if (!processingActive) {
        ImGui::Separator();
        ImGui::Spacing();
        renderParameters();
    }

    // ── Controles Start / Cancel ──────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();

    if (!processingActive) {
        if (!m_hasVideoInput) ImGui::BeginDisabled();
        if (ImGui::Button("Start Processing", ImVec2(140, 0))) {
            m_showResult = false;
            startProcessing();
        }
        if (!m_hasVideoInput) ImGui::EndDisabled();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.65f, 0.18f, 0.18f, 1.0f));
        if (ImGui::Button("Cancel Processing", ImVec2(140, 0)))
            cancelProcessing();
        ImGui::PopStyleColor();
    }
    ImGui::Spacing();

    // ── Progreso (solo mientras se procesa) ───────────────────────────────────
    if (processingActive) {
        int done  = m_processedFrames.load();
        int total = m_vidTotalFrames;
        int dets  = m_detectedTotal.load();

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.75f, 1.0f), "Procesando video...");
        ImGui::Spacing();

        float progress = (total > 0)
                         ? static_cast<float>(done) / static_cast<float>(total)
                         : 0.0f;
        char overlay[64];
        if (total > 0)
            std::snprintf(overlay, sizeof(overlay), "Frame %d / %d  (%d%%)",
                          done, total, static_cast<int>(progress * 100.0f));
        else
            std::snprintf(overlay, sizeof(overlay), "Frame %d procesado", done);

        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), overlay);
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
        ImGui::Text("Peatones detectados : %d", dets);

        double elapsed = ImGui::GetTime() - m_startTimeSec;
        ImGui::Text("Tiempo transcurrido : %.1f s", elapsed);

        if (done > 0 && total > 0 && elapsed > 0.5) {
            double rate      = static_cast<double>(done) / elapsed;
            double remaining = static_cast<double>(total - done) / std::max(rate, 0.01);
            ImGui::Text("Tiempo restante     : ~%.0f s", remaining);
        }
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    // ── Resultado ─────────────────────────────────────────────────────────────
    if (m_showResult) {
        ImGui::Separator();
        ImGui::Spacing();

        if (m_workerSuccess) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f),
                               "Procesamiento completado exitosamente.");
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
            ImGui::Text("Archivo guardado    : %s", m_workerOutputPath.c_str());
            ImGui::Text("Total peatones      : %d", m_detectedTotal.load());
            ImGui::Text("Tiempo total        : %.1f s", m_workerElapsedSec);
            ImGui::PopStyleColor();
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                               "Procesamiento finalizado.");
            ImGui::TextDisabled("%s", m_workerError.c_str());
        }
    }
}

// ─── Cargar metadatos del video (sin reproducir) ──────────────────────────────

void HOGPedestrianModule::loadVideoMeta(const std::string& path)
{
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "[HOGPedestrian] Cannot open video: %s\n", path.c_str());
        m_hasVideoInput = false;
        return;
    }

    m_videoInputPath = path;
    m_hasVideoInput  = true;

    m_vidFps = cap.get(cv::CAP_PROP_FPS);
    if (m_vidFps <= 0.0 || m_vidFps > 300.0) m_vidFps = 30.0;

    m_vidTotalFrames  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    m_vidWidth        = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_vidHeight       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_vidDurationSec  = (m_vidTotalFrames > 0) ? m_vidTotalFrames / m_vidFps : 0.0;
    cap.release();

    try {
        m_vidFileSizeBytes = fs::file_size(path);
    } catch (...) {
        m_vidFileSizeBytes = 0;
    }

    std::fprintf(stdout, "[HOGPedestrian] Video loaded: %s  (%dx%d @ %.1f fps, %d frames)\n",
                 path.c_str(), m_vidWidth, m_vidHeight, m_vidFps, m_vidTotalFrames);
}

// ─── Inicio del procesamiento ─────────────────────────────────────────────────

void HOGPedestrianModule::startProcessing()
{
    if (m_isProcessing.load() || !m_hasVideoInput) return;

    m_cancelRequested.store(false);
    m_processedFrames.store(0);
    m_detectedTotal.store(0);
    m_workerSuccess    = false;
    m_workerOutputPath.clear();
    m_workerError.clear();
    m_workerElapsedSec = 0.0f;
    m_startTimeSec     = ImGui::GetTime();

    m_isProcessing.store(true);
    // Thread creation is a synchronization point: worker sees all prior writes
    m_workerThread = std::thread([this]() { workerProcess(); });
}

void HOGPedestrianModule::cancelProcessing()
{
    m_cancelRequested.store(true);
}

// ─── Worker: procesamiento offline en thread separado ─────────────────────────

void HOGPedestrianModule::workerProcess()
{
    auto wallStart = std::chrono::steady_clock::now();

    // ── Abrir video de entrada ────────────────────────────────────────────────
    cv::VideoCapture cap(m_videoInputPath);
    if (!cap.isOpened()) {
        m_workerError = "No se pudo abrir el video de entrada.";
        m_isProcessing.store(false);
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0;
    int w  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // ── Construir ruta de salida ──────────────────────────────────────────────
    std::string outDir = m_videoOutputDir.empty() ? "exports" : m_videoOutputDir;
    try { fs::create_directories(outDir); } catch (...) {}

    std::string stem    = fs::path(m_videoInputPath).stem().string();
    std::string outPath = (fs::path(outDir) / (stem + "_hog_processed.avi")).string();

    // ── Abrir VideoWriter con codec MJPG (compatible sin ffmpeg) ─────────────
    cv::VideoWriter writer(outPath,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           fps,
                           cv::Size(w, h));
    if (!writer.isOpened()) {
        cap.release();
        m_workerError = "No se pudo crear el archivo de salida: " + outPath;
        m_isProcessing.store(false);
        return;
    }

    // ── Descriptor HOG local al thread (evita compartir m_hog) ───────────────
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // ── Snapshot de parámetros (leídos antes de iniciar el thread) ────────────
    double       hitThreshold  = m_hitThreshold;
    cv::Size     winStride(std::max(1, m_winStrideX), std::max(1, m_winStrideY));
    double       scale         = m_scale;
    double       groupThresh   = static_cast<double>(m_groupThreshold);

    // ── Procesamiento frame a frame ───────────────────────────────────────────
    cv::Mat frame;
    int totalDets = 0;

    while (!m_cancelRequested.load()) {
        if (!cap.read(frame) || frame.empty()) break;

        std::vector<cv::Rect> dets;
        hog.detectMultiScale(frame, dets, hitThreshold, winStride,
                             cv::Size(0, 0), scale, groupThresh);

        totalDets += static_cast<int>(dets.size());
        m_detectedTotal.store(totalDets);

        for (const auto& r : dets) {
            cv::rectangle(frame, r, cv::Scalar(0, 200, 0), 2);
            cv::rectangle(frame,
                          cv::Point(r.x + 4, r.y + 4),
                          cv::Point(r.x + r.width - 4, r.y + r.height - 4),
                          cv::Scalar(0, 230, 80), 1);
        }

        writer.write(frame);
        m_processedFrames.fetch_add(1);
    }

    writer.release();
    cap.release();

    auto wallEnd = std::chrono::steady_clock::now();
    m_workerElapsedSec = std::chrono::duration<float>(wallEnd - wallStart).count();

    if (m_cancelRequested.load()) {
        try { fs::remove(outPath); } catch (...) {}
        m_workerError   = "Procesamiento cancelado por el usuario.";
        m_workerSuccess = false;
    } else {
        m_workerOutputPath = outPath;
        m_workerSuccess    = true;
        std::fprintf(stdout, "[HOGPedestrian] Done: %s  (%d peatones, %.1f s)\n",
                     outPath.c_str(), totalDets, m_workerElapsedSec);
    }

    // Último write atómico: la UI lo detecta en pollWorker()
    m_isProcessing.store(false);
}

// ─── Polling del thread desde la UI ──────────────────────────────────────────

void HOGPedestrianModule::pollWorker()
{
    // Cuando el worker termina (m_isProcessing → false) y el thread sigue
    // joinable, hacemos join y mostramos el resultado.
    if (m_workerThread.joinable() && !m_isProcessing.load()) {
        m_workerThread.join();
        m_showResult = true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARÁMETROS HOG — compartidos por ambos modos
// ═══════════════════════════════════════════════════════════════════════════════

void HOGPedestrianModule::renderParameters()
{
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Parametros HOG + SVM");
    ImGui::Spacing();

    bool imgMode = (m_mode == HOGInputMode::Image);

    // Hit Threshold
    ImGui::Text("Hit Threshold (sensibilidad):");
    float ht = static_cast<float>(m_hitThreshold);
    if (ImGui::SliderFloat("##ht", &ht, -1.0f, 2.0f, "%.2f")) {
        m_hitThreshold = static_cast<double>(ht);
        if (imgMode && !m_imageBGR.empty()) m_imageNeedsDetect = true;
    }
    ImGui::TextDisabled("  Menor valor = mayor sensibilidad (mas falsos positivos).");
    ImGui::Spacing();

    // Win Stride
    ImGui::Text("Win Stride X (px):");
    if (ImGui::SliderInt("##wsx", &m_winStrideX, 4, 16))
        if (imgMode && !m_imageBGR.empty()) m_imageNeedsDetect = true;

    ImGui::Text("Win Stride Y (px):");
    if (ImGui::SliderInt("##wsy", &m_winStrideY, 4, 16))
        if (imgMode && !m_imageBGR.empty()) m_imageNeedsDetect = true;

    ImGui::TextDisabled("  Stride menor = busqueda mas densa = mas lento.");
    ImGui::Spacing();

    // Scale
    ImGui::Text("Scale (piramide multiescala):");
    float sc = static_cast<float>(m_scale);
    if (ImGui::SliderFloat("##sc", &sc, 1.01f, 1.5f, "%.3f")) {
        m_scale = static_cast<double>(sc);
        if (imgMode && !m_imageBGR.empty()) m_imageNeedsDetect = true;
    }
    ImGui::TextDisabled("  Scale menor = mas escalas = mas lento.");
    ImGui::Spacing();

    // Group Threshold
    ImGui::Text("Group Threshold (supresion de redundancias):");
    if (ImGui::SliderInt("##gt", &m_groupThreshold, 0, 10))
        if (imgMode && !m_imageBGR.empty()) m_imageNeedsDetect = true;

    ImGui::TextDisabled("  Mayor valor = menos bounding boxes finales.");
    ImGui::Spacing();
}
