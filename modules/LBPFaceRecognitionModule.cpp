// core/gl.h must be included before GLFW when GLFW_INCLUDE_NONE is active
#include "core/gl.h"
#include <GLFW/glfw3.h>

#include "imgui.h"

#include "LBPFaceRecognitionModule.h"
#include "core/FileDialog.h"
#include "core/ExportUtils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

// ─── Cascade candidate paths (tried in order) ────────────────────────────────

static const char* kCascadeCandidates[] = {
    // macOS — Homebrew Apple Silicon / Intel
    "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    // Linux
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    nullptr
};

// ─── Constructor / Destructor ────────────────────────────────────────────────

LBPFaceRecognitionModule::LBPFaceRecognitionModule()
{
    for (int i = 0; kCascadeCandidates[i]; ++i) {
        if (tryLoadCascade(kCascadeCandidates[i])) return;
    }
}

LBPFaceRecognitionModule::~LBPFaceRecognitionModule()
{
    Utils::deleteTexture(m_texRef);
    Utils::deleteTexture(m_texQuery);
    Utils::deleteTexture(m_texLBPRef);
    Utils::deleteTexture(m_texLBPQuery);
}

bool LBPFaceRecognitionModule::tryLoadCascade(const std::string& path)
{
    if (m_cascade.load(path)) {
        m_cascadeLoaded = true;
        m_cascadePath   = path;
        return true;
    }
    return false;
}

// ─── Image loading ────────────────────────────────────────────────────────────

void LBPFaceRecognitionModule::loadImage(int slot, const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) return;
    m_rawImg[slot]   = img;
    m_imgPath[slot]  = path;
    m_hasImage[slot] = true;
    m_hasFace[slot]  = false;
    m_hist[slot].clear();
    m_hasComparison  = false;
    m_needsProcess   = true;
}

// ─── Static LBP algorithms ───────────────────────────────────────────────────

// Circular LBP(R, P) with bilinear interpolation.
// Encodes the local texture into an 8-bit value (caps P at 8 for 1-byte codes).
cv::Mat LBPFaceRecognitionModule::computeLBP(const cv::Mat& gray,
                                              int radius, int neighbors)
{
    const int P = std::min(neighbors, 8);
    cv::Mat result = cv::Mat::zeros(gray.size(), CV_8UC1);

    for (int y = radius; y < gray.rows - radius; ++y) {
        for (int x = radius; x < gray.cols - radius; ++x) {
            float center = static_cast<float>(gray.at<uchar>(y, x));
            uchar code   = 0;
            for (int n = 0; n < P; ++n) {
                // Sample point on the circle (counter-clockwise from the right)
                double angle = 2.0 * CV_PI * n / P;
                double nx    = x + radius * std::cos(angle);
                double ny    = y - radius * std::sin(angle);

                // Bilinear interpolation
                int    x0 = static_cast<int>(nx);
                int    y0 = static_cast<int>(ny);
                double dx  = nx - x0;
                double dy  = ny - y0;
                int    x1  = std::min(x0 + 1, gray.cols - 1);
                int    y1  = std::min(y0 + 1, gray.rows - 1);
                x0 = std::max(0, x0);
                y0 = std::max(0, y0);

                double val = (1.0 - dx) * (1.0 - dy) * gray.at<uchar>(y0, x0)
                           +        dx  * (1.0 - dy) * gray.at<uchar>(y0, x1)
                           + (1.0 - dx) *        dy  * gray.at<uchar>(y1, x0)
                           +        dx  *        dy  * gray.at<uchar>(y1, x1);

                if (val >= center) code |= static_cast<uchar>(1 << n);
            }
            result.at<uchar>(y, x) = code;
        }
    }
    return result;
}

// LBPH: divide lbp into a gridN×gridN spatial grid, compute a numBins-bin
// histogram per cell, normalize by cell area, and concatenate.
std::vector<float> LBPFaceRecognitionModule::computeLBPHistogram(
    const cv::Mat& lbp, int gridN, int numBins)
{
    std::vector<float> feature;
    feature.reserve(gridN * gridN * numBins);

    int cellW = lbp.cols / gridN;
    int cellH = lbp.rows / gridN;

    for (int gy = 0; gy < gridN; ++gy) {
        for (int gx = 0; gx < gridN; ++gx) {
            int x0 = gx * cellW;
            int y0 = gy * cellH;
            int x1 = (gx == gridN - 1) ? lbp.cols : x0 + cellW;
            int y1 = (gy == gridN - 1) ? lbp.rows : y0 + cellH;

            cv::Mat cell = lbp(cv::Rect(x0, y0, x1 - x0, y1 - y0));
            float total  = static_cast<float>((x1 - x0) * (y1 - y0));

            std::vector<float> hist(numBins, 0.0f);
            for (int r = 0; r < cell.rows; ++r)
                for (int c = 0; c < cell.cols; ++c)
                    hist[cell.at<uchar>(r, c) % numBins] += 1.0f;

            for (auto& v : hist) v /= total;  // normalize per cell area
            feature.insert(feature.end(), hist.begin(), hist.end());
        }
    }
    return feature;
}

// Symmetric chi-square distance (Pele & Werman variant, epsilon-safe).
float LBPFaceRecognitionModule::chiSquare(const std::vector<float>& h1,
                                           const std::vector<float>& h2)
{
    float d = 0.0f;
    for (size_t i = 0; i < h1.size(); ++i) {
        float a = h1[i], b = h2[i];
        float s = a + b;
        if (s > 1e-7f) d += (a - b) * (a - b) / s;
    }
    return d;
}

// ─── Per-slot processing ──────────────────────────────────────────────────────

void LBPFaceRecognitionModule::processSlot(int slot)
{
    if (!m_hasImage[slot] || m_rawImg[slot].empty()) return;

    cv::Mat gray;
    cv::cvtColor(m_rawImg[slot], gray, cv::COLOR_BGR2GRAY);

    // Equalize histogram before Haar detection for robustness under lighting
    cv::Mat grayEq;
    cv::equalizeHist(gray, grayEq);

    // ── Face detection ────────────────────────────────────────────────────────
    m_hasFace[slot] = false;
    cv::Rect faceRect(0, 0, m_rawImg[slot].cols, m_rawImg[slot].rows);

    if (m_cascadeLoaded) {
        std::vector<cv::Rect> faces;
        m_cascade.detectMultiScale(grayEq, faces,
                                    m_detectionScale, m_minNeighbors,
                                    0, cv::Size(30, 30));
        if (!faces.empty()) {
            // Pick the largest detected face
            auto largest = std::max_element(faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b) {
                    return a.area() < b.area();
                });
            faceRect        = *largest;
            m_hasFace[slot] = true;
        }
    }

    // Clamp to image bounds
    faceRect &= cv::Rect(0, 0, m_rawImg[slot].cols, m_rawImg[slot].rows);
    if (faceRect.empty())
        faceRect = cv::Rect(0, 0, m_rawImg[slot].cols, m_rawImg[slot].rows);

    m_faceRect[slot] = faceRect;

    // ── Annotated display image ───────────────────────────────────────────────
    cv::Mat display = m_rawImg[slot].clone();
    if (m_hasFace[slot]) {
        cv::rectangle(display, faceRect, cv::Scalar(50, 220, 50), 2);
        cv::putText(display, "Rostro", faceRect.tl() + cv::Point(4, -6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 220, 50), 1,
                    cv::LINE_AA);
    } else if (m_cascadeLoaded) {
        // Warn: no face found — full image used as ROI
        cv::putText(display, "Sin rostro detectado",
                    cv::Point(8, display.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 100, 255), 1,
                    cv::LINE_AA);
    }

    // Scale display to ≤ 700px
    const int kMaxDisplayDim = 700;
    float dScale = std::min(1.0f,
        static_cast<float>(kMaxDisplayDim) / static_cast<float>(std::max(display.cols, display.rows)));
    cv::Mat displaySmall;
    if (dScale < 0.999f) cv::resize(display, displaySmall, cv::Size(), dScale, dScale, cv::INTER_AREA);
    else                 displaySmall = display;

    // ── LBP computation on 128×128 face crop ─────────────────────────────────
    cv::Mat faceGray = gray(faceRect);
    cv::Mat face128;
    cv::resize(faceGray, face128, cv::Size(128, 128), 0, 0, cv::INTER_AREA);
    m_faceROI[slot] = face128;

    const int numBins = (m_neighbors <= 4) ? 16 : 256;
    cv::Mat lbp       = computeLBP(face128, m_radius, m_neighbors);
    m_hist[slot]      = computeLBPHistogram(lbp, m_gridN, numBins);

    // ── LBP texture display (256×256) with grid overlay ──────────────────────
    cv::Mat lbpBGR;
    cv::cvtColor(lbp, lbpBGR, cv::COLOR_GRAY2BGR);

    cv::Mat lbpDisplay;
    cv::resize(lbpBGR, lbpDisplay, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);

    // Draw grid lines to visualize histogram cells
    int cellW = lbpDisplay.cols / m_gridN;
    int cellH = lbpDisplay.rows / m_gridN;
    for (int i = 1; i < m_gridN; ++i) {
        cv::line(lbpDisplay, cv::Point(i * cellW, 0),
                              cv::Point(i * cellW, lbpDisplay.rows),
                              cv::Scalar(0, 200, 0), 1);
        cv::line(lbpDisplay, cv::Point(0, i * cellH),
                              cv::Point(lbpDisplay.cols, i * cellH),
                              cv::Scalar(0, 200, 0), 1);
    }

    // Cache for export and upload to GPU
    if (slot == 0) {
        m_displayRef    = displaySmall;
        m_lbpDisplayRef = lbpDisplay;
        Utils::updateTexture(m_texRef,    displaySmall);
        Utils::updateTexture(m_texLBPRef, lbpDisplay);
    } else {
        m_displayQuery    = displaySmall;
        m_lbpDisplayQuery = lbpDisplay;
        Utils::updateTexture(m_texQuery,    displaySmall);
        Utils::updateTexture(m_texLBPQuery, lbpDisplay);
    }
}

// ─── Comparison ───────────────────────────────────────────────────────────────

void LBPFaceRecognitionModule::compareHistograms()
{
    m_hasComparison = false;
    if (m_hist[0].empty() || m_hist[1].empty()) return;
    if (m_hist[0].size() != m_hist[1].size())   return;

    m_chiDist     = chiSquare(m_hist[0], m_hist[1]);
    m_chiDistNorm = m_chiDist / static_cast<float>(m_gridN * m_gridN);
    m_hasComparison = true;
}

// ─── Reprocess ────────────────────────────────────────────────────────────────

void LBPFaceRecognitionModule::reprocess()
{
    auto t0 = std::chrono::high_resolution_clock::now();

    if (m_hasImage[0]) processSlot(0);
    if (m_hasImage[1]) processSlot(1);
    if (m_hasImage[0] && m_hasImage[1]) compareHistograms();

    auto t1         = std::chrono::high_resolution_clock::now();
    m_processTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    m_needsProcess  = false;
}

// ─── Export ───────────────────────────────────────────────────────────────────

void LBPFaceRecognitionModule::exportSlot(int slot)
{
    const cv::Mat& disp = (slot == 0) ? m_displayRef    : m_displayQuery;
    const cv::Mat& lbp  = (slot == 0) ? m_lbpDisplayRef : m_lbpDisplayQuery;
    if (disp.empty() && lbp.empty()) return;

    std::string label = (slot == 0) ? "ref" : "query";
    if (!disp.empty()) {
        std::string p = ExportUtils::nextExportPath("exports",
                            "lbp_face_" + label, "png");
        cv::imwrite(p, disp);
        m_exportMsg     = "Exportado: " + p;
        m_exportMsgTime = glfwGetTime();
    }
    if (!lbp.empty()) {
        std::string p = ExportUtils::nextExportPath("exports",
                            "lbp_texture_" + label, "png");
        cv::imwrite(p, lbp);
    }
}

// ─── Reset ────────────────────────────────────────────────────────────────────

void LBPFaceRecognitionModule::reset()
{
    Utils::deleteTexture(m_texRef);
    Utils::deleteTexture(m_texQuery);
    Utils::deleteTexture(m_texLBPRef);
    Utils::deleteTexture(m_texLBPQuery);

    for (int i = 0; i < 2; ++i) {
        m_rawImg[i]    = cv::Mat();
        m_faceROI[i]   = cv::Mat();
        m_hist[i].clear();
        m_hasImage[i]  = false;
        m_hasFace[i]   = false;
        m_imgPath[i].clear();
    }
    m_displayRef    = m_displayQuery    = cv::Mat();
    m_lbpDisplayRef = m_lbpDisplayQuery = cv::Mat();

    m_hasComparison = false;
    m_chiDist = m_chiDistNorm = 0.0f;
    m_needsProcess = false;
    m_processTimeMs = 0.0f;
    m_exportMsg.clear();
}

// ─── UI ──────────────────────────────────────────────────────────────────────

void LBPFaceRecognitionModule::renderUI()
{
    ImGui::PushID("LBPFaceRecognition");

    if (m_needsProcess) reprocess();

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.75f, 1.0f), "LBP Face Recognition");
    ImGui::TextDisabled(
        "Descripcion de texturas faciales con Local Binary Patterns — clasico y eficiente.");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Cascade status ────────────────────────────────────────────────────────
    if (!m_cascadeLoaded) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
            "Clasificador Haar no encontrado. Usa el boton de abajo para cargarlo.");
        if (ImGui::Button("Browse Cascade File (.xml)##lbp", ImVec2(240, 0))) {
            // reuse openImage for .xml — user selects manually
            auto p = FileDialog::openImage();
            if (!p.empty()) tryLoadCascade(p);
        }
        ImGui::Spacing();
    } else {
        ImGui::TextDisabled("Cascade: %s", m_cascadePath.c_str());
    }

    // ── Load / Reset buttons ──────────────────────────────────────────────────
    if (ImGui::Button("Ref. Face##lbp", ImVec2(140, 0))) {
        auto p = FileDialog::openImage();
        if (!p.empty()) loadImage(0, p);
    }
    ImGui::SameLine();
    if (ImGui::Button("Comparison Face##lbp", ImVec2(155, 0))) {
        auto p = FileDialog::openImage();
        if (!p.empty()) loadImage(1, p);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset##lbp", ImVec2(60, 0)))
        reset();

    if (!m_imgPath[0].empty()) ImGui::TextDisabled("  Ref: %s", m_imgPath[0].c_str());
    if (!m_imgPath[1].empty()) ImGui::TextDisabled("  Cmp: %s", m_imgPath[1].c_str());

    ImGui::Spacing();

    // ── Action buttons ────────────────────────────────────────────────────────
    bool anyLoaded = m_hasImage[0] || m_hasImage[1];
    bool canExport = m_hasImage[0] || m_hasImage[1];

    if (!anyLoaded) ImGui::BeginDisabled();
    if (ImGui::Button("Process##lbp", ImVec2(100, 0))) {
        reprocess();
    }
    if (!anyLoaded) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!m_hasImage[0]) ImGui::BeginDisabled();
    if (ImGui::Button("Export Ref##lbp", ImVec2(100, 0)))  exportSlot(0);
    if (!m_hasImage[0]) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!m_hasImage[1]) ImGui::BeginDisabled();
    if (ImGui::Button("Export Cmp##lbp", ImVec2(100, 0)))  exportSlot(1);
    if (!m_hasImage[1]) ImGui::EndDisabled();

    if (!m_exportMsg.empty()) {
        if (glfwGetTime() - m_exportMsgTime < 3.0)
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "%s", m_exportMsg.c_str());
        else
            m_exportMsg.clear();
    }

    ImGui::Spacing();
    ImGui::Separator();

    // ── Parameters ────────────────────────────────────────────────────────────
    ImGui::Text("Parametros LBP:");
    ImGui::Spacing();

    bool changed = false;

    ImGui::SetNextItemWidth(160);
    changed |= ImGui::SliderInt("Grid Size (NxN)##lbp", &m_gridN, 2, 16);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Grid de celdas para el histograma espacial.\n"
                          "Grid = 8 => 8x8 celdas, cada una con su histograma LBP.");
    ImGui::SameLine(0, 24);
    ImGui::SetNextItemWidth(160);
    changed |= ImGui::SliderInt("Radius##lbp", &m_radius, 1, 3);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Radio del circulo de muestreo LBP(R, P).\n"
                          "Radio mayor = captura mas contexto de textura.");

    ImGui::SetNextItemWidth(160);
    // Neighbors: only 4 or 8 are meaningful with 8-bit code output
    int neighborsIdx = (m_neighbors <= 4) ? 0 : 1;
    const char* neighborOpts[] = {"4 (16 bins)", "8 (256 bins)"};
    if (ImGui::Combo("Neighbors##lbp", &neighborsIdx, neighborOpts, 2)) {
        m_neighbors = (neighborsIdx == 0) ? 4 : 8;
        changed = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Puntos de muestreo en el circulo. P=8 => 256 codigos posibles.\n"
                          "P=4 => 16 codigos (mayor velocidad, menor precision).");

    ImGui::SameLine(0, 24);
    ImGui::SetNextItemWidth(160);
    changed |= ImGui::SliderFloat("Detect Scale##lbp", &m_detectionScale, 1.05f, 1.5f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Factor de escala de la piramide Haar.\n"
                          "Menor = mas lento pero detecta mas tallas.");

    ImGui::SetNextItemWidth(160);
    changed |= ImGui::SliderInt("Min Neighbors##lbp", &m_minNeighbors, 1, 8);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Umbral de vecinos coincidentes en Haar.\n"
                          "Menor = detecta mas (mas falsos positivos).");

    ImGui::SameLine(0, 24);
    ImGui::SetNextItemWidth(160);
    changed |= ImGui::SliderFloat("Same-Person Threshold##lbp",
                                   &m_threshold, 0.5f, 10.0f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Umbral de distancia chi-cuadrado normalizada.\n"
                          "Distancia < umbral => mismo sujeto.");

    if (changed && anyLoaded)
        m_needsProcess = true;

    ImGui::Spacing();
    ImGui::Separator();

    // ── Statistics & comparison result ────────────────────────────────────────
    if (m_hasImage[0] || m_hasImage[1]) {
        ImGui::Text("Estadisticas:");
        ImGui::Spacing();

        for (int s = 0; s < 2; ++s) {
            if (!m_hasImage[s]) continue;
            const char* label = (s == 0) ? "Referencia" : "Comparacion";
            if (m_hasFace[s])
                ImGui::Text("%s:  %d x %d px   |   Rostro: %dx%d @ (%d,%d)",
                    label,
                    m_rawImg[s].cols, m_rawImg[s].rows,
                    m_faceRect[s].width, m_faceRect[s].height,
                    m_faceRect[s].x, m_faceRect[s].y);
            else
                ImGui::Text("%s:  %d x %d px   |   Sin rostro — usando imagen completa",
                    label, m_rawImg[s].cols, m_rawImg[s].rows);
        }

        if (m_hasComparison) {
            ImGui::Spacing();
            ImGui::Text("Distancia chi-cuadrado: %.4f  (normalizada: %.4f)",
                        m_chiDist, m_chiDistNorm);

            ImGui::Spacing();
            bool sameP = (m_chiDistNorm < m_threshold);
            if (sameP)
                ImGui::TextColored(ImVec4(0.20f, 1.00f, 0.30f, 1.0f),
                    "MISMO SUJETO  (dist. %.3f < umbral %.3f)",
                    m_chiDistNorm, m_threshold);
            else
                ImGui::TextColored(ImVec4(1.00f, 0.40f, 0.30f, 1.0f),
                    "SUJETO DIFERENTE  (dist. %.3f >= umbral %.3f)",
                    m_chiDistNorm, m_threshold);

            // Visual similarity bar
            ImGui::Spacing();
            float bar = std::max(0.0f, std::min(1.0f, 1.0f - m_chiDistNorm / (m_threshold * 3.0f)));
            ImGui::TextDisabled("Similitud aproximada:");
            ImGui::SameLine();
            char overlay[32];
            std::snprintf(overlay, sizeof(overlay), "%.0f%%", bar * 100.0f);
            ImGui::ProgressBar(bar, ImVec2(200.0f, 0.0f), overlay);
        }

        if (m_processTimeMs > 0.0f)
            ImGui::TextDisabled("Tiempo total de procesamiento: %.1f ms", m_processTimeMs);

        ImGui::Spacing();
        ImGui::Separator();
    }

    // ── Academic explanation ──────────────────────────────────────────────────
    ImGui::Spacing();
    if (ImGui::TreeNode("Acerca de LBP##tree")) {
        ImGui::Spacing();
        ImGui::TextWrapped(
            "LBP describe texturas locales utilizando comparaciones binarias entre "
            "pixeles vecinos. Cada pixel se codifica segun si sus vecinos son "
            "mayores o iguales al pixel central, generando un patron binario de P bits.");
        ImGui::Spacing();
        ImGui::BulletText("Rapido: opera directamente sobre intensidades de gris.");
        ImGui::BulletText("Sin entrenamiento: el descriptor es puramente geometrico.");
        ImGui::BulletText("Sensible a iluminacion extrema: no es totalmente invariante.");
        ImGui::BulletText("Sensible a pose: el rostro debe estar aproximadamente frontal.");
        ImGui::BulletText("Menor precision que DL: metodos deep learning superan ampliamente.");
        ImGui::Spacing();
        ImGui::TextWrapped(
            "LBPH (LBP Histograms): divide la imagen en una cuadricula NxN, computa "
            "el histograma LBP de cada celda y concatena. La comparacion usa "
            "distancia chi-cuadrado entre los vectores de caracteristicas.");
        ImGui::Spacing();
        ImGui::TextDisabled("Referencia: Ojala et al., \"A Comparative Study of Texture");
        ImGui::TextDisabled("Measures with Classification Based on Feature Distributions\",");
        ImGui::TextDisabled("Pattern Recognition, 1996.");
        ImGui::TreePop();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ── Image display ─────────────────────────────────────────────────────────
    float contentW = ImGui::GetContentRegionAvail().x;

    if (!m_hasImage[0] && !m_hasImage[1]) {
        ImGui::TextDisabled(
            "Carga una imagen de referencia y una de comparacion para comenzar.");
        ImGui::PopID();
        return;
    }

    // Row 1: annotated originals side by side
    float panelW = (contentW - 16.0f) * 0.5f;
    float maxH   = 300.0f;

    ImGui::TextDisabled("Imagenes originales con deteccion de rostro:");
    ImGui::Spacing();

    ImGui::BeginGroup();
    ImGui::TextDisabled("Referencia%s", m_hasFace[0] ? "  (rostro detectado)" : "");
    if (m_texRef.valid()) {
        float s = std::min(panelW / static_cast<float>(m_texRef.width),
                           maxH   / static_cast<float>(m_texRef.height));
        ImGui::Image((ImTextureID)(uintptr_t)m_texRef.id,
                     ImVec2(m_texRef.width * s, m_texRef.height * s));
    } else if (m_hasImage[0]) {
        ImGui::TextDisabled("(procesando...)");
    }
    ImGui::EndGroup();

    ImGui::SameLine(0, 16);

    ImGui::BeginGroup();
    ImGui::TextDisabled("Comparacion%s", m_hasFace[1] ? "  (rostro detectado)" : "");
    if (m_texQuery.valid()) {
        float s = std::min(panelW / static_cast<float>(m_texQuery.width),
                           maxH   / static_cast<float>(m_texQuery.height));
        ImGui::Image((ImTextureID)(uintptr_t)m_texQuery.id,
                     ImVec2(m_texQuery.width * s, m_texQuery.height * s));
    } else if (m_hasImage[1]) {
        ImGui::TextDisabled("(procesando...)");
    }
    ImGui::EndGroup();

    // Row 2: LBP textures side by side (fixed 256×256 with grid overlay)
    if (m_texLBPRef.valid() || m_texLBPQuery.valid()) {
        ImGui::Spacing();
        ImGui::TextDisabled("Texturas LBP (grid de celdas en verde):");
        ImGui::Spacing();

        ImGui::BeginGroup();
        ImGui::TextDisabled("LBP Referencia  (R=%d, P=%d, Grid=%dx%d)",
                            m_radius, std::min(m_neighbors, 8), m_gridN, m_gridN);
        if (m_texLBPRef.valid()) {
            float s = std::min(panelW / static_cast<float>(m_texLBPRef.width),
                               maxH   / static_cast<float>(m_texLBPRef.height));
            ImGui::Image((ImTextureID)(uintptr_t)m_texLBPRef.id,
                         ImVec2(m_texLBPRef.width * s, m_texLBPRef.height * s));
        }
        ImGui::EndGroup();

        ImGui::SameLine(0, 16);

        ImGui::BeginGroup();
        ImGui::TextDisabled("LBP Comparacion  (R=%d, P=%d, Grid=%dx%d)",
                            m_radius, std::min(m_neighbors, 8), m_gridN, m_gridN);
        if (m_texLBPQuery.valid()) {
            float s = std::min(panelW / static_cast<float>(m_texLBPQuery.width),
                               maxH   / static_cast<float>(m_texLBPQuery.height));
            ImGui::Image((ImTextureID)(uintptr_t)m_texLBPQuery.id,
                         ImVec2(m_texLBPQuery.width * s, m_texLBPQuery.height * s));
        }
        ImGui::EndGroup();
    }

    ImGui::PopID();
}
