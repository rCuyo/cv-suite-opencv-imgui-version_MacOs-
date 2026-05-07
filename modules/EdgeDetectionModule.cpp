#include "EdgeDetectionModule.h"
#include "core/ImageLoader.h"
#include "core/FileDialog.h"
#include "core/ExportUtils.h"
#include "core/HistogramUtils.h"
#include "core/HistogramRenderer.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

EdgeDetectionModule::~EdgeDetectionModule()
{
    Utils::deleteTexture(m_texOrig);
    Utils::deleteTexture(m_texEdges);
    Utils::deleteTexture(m_texContours);
}

// ─── Top-level UI ─────────────────────────────────────────────────────────────

void EdgeDetectionModule::renderUI()
{
    ImGui::PushID("EdgeDetectionModule");

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Detección de bordes");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Mode selector ─────────────────────────────────────────────────────────
    ImGui::Text("Modo:");
    ImGui::SameLine();
    auto modeBtn = [&](const char* label, EdgeMode mode) {
        bool sel = (m_mode == mode);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::Button(label)) { m_mode = mode; m_needsProcess = true; }
        if (sel) ImGui::PopStyleColor();
        ImGui::SameLine();
    };
    modeBtn("OCR / Document", EdgeMode::OCRDocument);
    modeBtn("Segmentation",   EdgeMode::Segmentation);
    ImGui::NewLine();

    // ── File loader ───────────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();
    if (ImGui::Button("Explorar...", ImVec2(90, 0))) {
        std::string p = FileDialog::openImage();
        if (!p.empty()) loadAndProcess(p);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_loadedPath.empty() ? "Sin imagen cargada" : m_loadedPath.c_str());
    ImGui::Spacing();

    // ── Canny parameter controls ──────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();

    // ── Gaussian Blur ─────────────────────────────────────────────────────────
    // The image is convolved with a Gaussian kernel before computing gradients.
    // This low-pass filter suppresses high-frequency noise so that every steep
    // intensity change corresponds to a real structural edge, not sensor noise.
    // A larger kernel blurs more but may merge nearby edges.
    ImGui::Text("Kernel Gaussian de pre-desenfoque (impar, >= 3):");
    if (ImGui::SliderInt("##gk", &m_gaussKernel, 3, 21))
    {
        if (m_gaussKernel % 2 == 0) m_gaussKernel++;
        m_needsProcess = true;
    }

    // ── Sobel Gradient ────────────────────────────────────────────────────────
    // Canny runs two 1-D Sobel operators (horizontal + vertical) to estimate the
    // first derivative of intensity. Magnitude G = sqrt(Gx²+Gy²) measures edge
    // strength; direction θ = atan2(Gy,Gx) shows the edge orientation.
    // Aperture 3 is standard; 5 or 7 give smoother gradients at higher cost.
    const char* apertures[] = { "3", "5", "7" };
    int apertureIdx = (m_apertureSize - 3) / 2;
    ImGui::Text("Tamaño de apertura Sobel:");
    if (ImGui::Combo("##ap", &apertureIdx, apertures, 3))
    {
        m_apertureSize = apertureIdx * 2 + 3;
        m_needsProcess = true;
    }

    // ── Hysteresis thresholds ─────────────────────────────────────────────────
    // After non-maximum suppression thins ridges to 1-pixel-wide candidates,
    // hysteresis classifies them:
    //   G > high  → definite edge (always kept)
    //   low < G ≤ high → weak edge (kept only if 8-connected to a definite edge)
    //   G ≤ low   → discarded
    // Two thresholds eliminate isolated noise while preserving continuous edges.
    ImGui::Text("Umbral inferior Canny:");
    if (ImGui::SliderInt("##cl", &m_cannyLow, 1, m_cannyHigh - 1))
        m_needsProcess = true;

    ImGui::Text("Umbral superior Canny:");
    if (ImGui::SliderInt("##ch", &m_cannyHigh, m_cannyLow + 1, 500))
        m_needsProcess = true;

    // ── Segmentation-only controls ────────────────────────────────────────────
    if (m_mode == EdgeMode::Segmentation)
    {
        ImGui::Spacing();
        ImGui::Text("Área mínima de contorno (px²):");
        if (ImGui::SliderInt("##mca", &m_minContourArea, 0, 5000))
            m_needsProcess = true;
    }

    // ── Lazy processing ───────────────────────────────────────────────────────
    if (m_needsProcess && !m_gray.empty()) {
        reprocess();
        m_needsProcess = false;
    }

    // ── Image viewer ──────────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawImageRow();

    // ── Export ────────────────────────────────────────────────────────────────
    if (m_texEdges.valid()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Exportar resultado", ImVec2(140, 0))) {
            cv::Mat toExport = m_texContours.valid() ? m_contourViz : m_edges;
            std::string path = ExportUtils::nextExportPath("exports", "canny_resultado");
            if (cv::imwrite(path, toExport))
                std::fprintf(stdout, "[Edge] Exportado: %s\n", path.c_str());
        }

        ImGui::SameLine();

        if (ImGui::Button("Exportar histogramas", ImVec2(150, 0))) {
            auto histOrig  = HistogramUtils::compute(m_gray);
            auto histEdges = HistogramUtils::compute(m_edges);

            HistogramRenderer::renderAndSave(
                histOrig,
                ExportUtils::nextExportPath("exports", "canny_histograma_original"),
                "Original");

            HistogramRenderer::renderAndSave(
                histEdges,
                ExportUtils::nextExportPath("exports", "canny_histograma_bordes"),
                "Bordes Canny");
        }
    }

    ImGui::PopID();
}

// ─── Load & process ───────────────────────────────────────────────────────────

void EdgeDetectionModule::loadAndProcess(const std::string& path)
{
    m_original = ImageLoader::load(path);
    if (m_original.empty()) {
        std::fprintf(stderr, "[EdgeDetection] Failed to load: %s\n", path.c_str());
        return;
    }
    m_loadedPath = path;
    cv::cvtColor(m_original, m_gray, cv::COLOR_BGR2GRAY);
    Utils::updateTexture(m_texOrig, m_original);
    m_needsProcess = true;
}

// ─── Processing ───────────────────────────────────────────────────────────────

void EdgeDetectionModule::reprocess()
{
    if (m_gray.empty()) return;

    switch (m_mode)
    {
    case EdgeMode::OCRDocument:  processOCR(m_gray);          break;
    case EdgeMode::Segmentation: processSegmentation(m_gray); break;
    }
}

// ── OCR / document ────────────────────────────────────────────────────────────

void EdgeDetectionModule::processOCR(const cv::Mat& gray)
{
    cv::Mat blurred;
    int ksize = m_gaussKernel;
    if (ksize < 3)      ksize = 3;
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), 0);

    cv::Canny(blurred, m_edges, m_cannyLow, m_cannyHigh, m_apertureSize);

    Utils::updateTexture(m_texEdges, m_edges);
    Utils::deleteTexture(m_texContours);
}

// ── Segmentation ──────────────────────────────────────────────────────────────

void EdgeDetectionModule::processSegmentation(const cv::Mat& gray)
{
    cv::Mat blurred;
    int ksize = m_gaussKernel;
    if (ksize < 3)      ksize = 3;
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), 0);
    cv::Canny(blurred, m_edges, m_cannyLow, m_cannyHigh, m_apertureSize);

    Utils::updateTexture(m_texEdges, m_edges);

    // RETR_EXTERNAL: only outermost contours (faster, no nested shapes).
    // CHAIN_APPROX_SIMPLE: compress collinear segments to endpoints only.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(m_edges.clone(), contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::cvtColor(gray, m_contourViz, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < static_cast<int>(contours.size()); ++i)
    {
        if (cv::contourArea(contours[i]) < m_minContourArea) continue;

        cv::Scalar colour(
            (i * 47  + 80) % 200 + 55,
            (i * 113 + 40) % 200 + 55,
            (i * 211 + 60) % 200 + 55);

        cv::drawContours(m_contourViz, contours, i, colour, 2);

        cv::Rect bbox = cv::boundingRect(contours[i]);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f px2", cv::contourArea(contours[i]));
        cv::putText(m_contourViz, buf,
                    cv::Point(bbox.x, bbox.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, colour, 1);
    }

    Utils::updateTexture(m_texContours, m_contourViz);
}

// ─── Image row renderer ───────────────────────────────────────────────────────

void EdgeDetectionModule::drawImageRow()
{
    if (!m_texOrig.valid()) {
        ImGui::TextDisabled("Sin imagen cargada. Haz clic en Explorar para abrir una imagen.");
        return;
    }

    bool hasContours = m_texContours.valid();
    int  cols        = hasContours ? 3 : 2;

    float avail   = ImGui::GetContentRegionAvail().x;
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float imgW    = (avail - spacing * (cols - 1)) / static_cast<float>(cols);
    float aspect  = static_cast<float>(m_texOrig.height) / static_cast<float>(m_texOrig.width);
    float imgH    = imgW * aspect;

    ImGui::Text("Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("Bordes Canny");
    if (hasContours) {
        ImGui::SameLine(2.0f * (imgW + spacing));
        ImGui::Text("Contornos");
    }

    ImGui::Image((ImTextureID)(uintptr_t)m_texOrig.id,   ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texEdges.id,  ImVec2(imgW, imgH));
    if (hasContours) {
        ImGui::SameLine();
        ImGui::Image((ImTextureID)(uintptr_t)m_texContours.id, ImVec2(imgW, imgH));
    }
}
