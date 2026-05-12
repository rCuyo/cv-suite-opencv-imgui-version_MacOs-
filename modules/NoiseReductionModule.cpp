#include "NoiseReductionModule.h"
#include "core/ImageLoader.h"
#include "core/FileDialog.h"
#include "core/HistogramUtils.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <algorithm>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

NoiseReductionModule::~NoiseReductionModule()
{
    Utils::deleteTexture(m_texOrig);
    Utils::deleteTexture(m_texResult);
}

// ─── Top-level UI ─────────────────────────────────────────────────────────────

void NoiseReductionModule::renderUI()
{
    ImGui::PushID("NoiseReductionModule");

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.8f, 0.4f, 1.0f, 1.0f), "Noise Reduction & Filters");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Mode selector ─────────────────────────────────────────────────────────
    ImGui::Text("Filter:");
    ImGui::SameLine();
    auto modeBtn = [&](const char* label, NoiseFilterMode mode) {
        bool sel = (m_mode == mode);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::Button(label)) { m_mode = mode; m_needsProcess = true; }
        if (sel) ImGui::PopStyleColor();
        ImGui::SameLine();
    };
    modeBtn("Gaussian",  NoiseFilterMode::Gaussian);
    modeBtn("Median",    NoiseFilterMode::Median);
    modeBtn("Bilateral", NoiseFilterMode::Bilateral);
    ImGui::NewLine();
    ImGui::Spacing();

    // ── File loader ───────────────────────────────────────────────────────────
    if (ImGui::Button("Browse...", ImVec2(80, 0))) {
        std::string p = FileDialog::openImage();
        if (!p.empty()) loadAndProcess(p);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_loadedPath.empty() ? "No image loaded." : m_loadedPath.c_str());
    ImGui::Spacing();

    // ── Mode-specific controls ────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();

    switch (m_mode)
    {
    // ── Gaussian ──────────────────────────────────────────────────────────────
    case NoiseFilterMode::Gaussian:
    {
        ImGui::TextDisabled("Filtro lineal basado en convolucion.");
        ImGui::TextDisabled("Gaussian Blur suaviza la imagen reduciendo ruido de alta frecuencia.");
        ImGui::Spacing();

        ImGui::Text("Kernel size (odd):");
        if (ImGui::SliderInt("##gk", &m_gaussKernel, 1, 31)) {
            if (m_gaussKernel % 2 == 0) m_gaussKernel++;
            m_needsProcess = true;
        }
        ImGui::Text("Sigma  (0 = auto from kernel):");
        if (ImGui::SliderFloat("##gs", &m_gaussSigma, 0.0f, 10.0f, "%.1f"))
            m_needsProcess = true;

        ImGui::Spacing();
        if (ImGui::Button("Reset##g")) {
            m_gaussKernel  = 5;
            m_gaussSigma   = 0.0f;
            m_needsProcess = true;
        }
        break;
    }

    // ── Median ────────────────────────────────────────────────────────────────
    case NoiseFilterMode::Median:
    {
        ImGui::TextDisabled("Filtro estadistico basado en mediana local.");
        ImGui::TextDisabled("Median Blur elimina ruido preservando mejor los bordes.");
        ImGui::Spacing();
        ImGui::TextDisabled("El ruido sal y pimienta consiste en pixeles blancos y");
        ImGui::TextDisabled("negros aleatorios distribuidos sobre la imagen.");
        ImGui::Spacing();

        ImGui::Text("Kernel size (odd):");
        if (ImGui::SliderInt("##mk", &m_medianKernel, 1, 31)) {
            if (m_medianKernel % 2 == 0) m_medianKernel++;
            m_needsProcess = true;
        }

        ImGui::Spacing();
        if (ImGui::Button("Reset##m")) {
            m_medianKernel = 5;
            m_needsProcess = true;
        }
        break;
    }

    // ── Bilateral ─────────────────────────────────────────────────────────────
    case NoiseFilterMode::Bilateral:
    {
        ImGui::TextDisabled("Filtro no lineal que preserva bordes.");
        ImGui::TextDisabled("Bilateral Filter reduce ruido manteniendo bordes importantes.");
        ImGui::Spacing();

        ImGui::Text("Diameter:");
        if (ImGui::SliderInt("##bd", &m_bilateralD, 1, 25))
            m_needsProcess = true;

        ImGui::Text("Sigma Color  (rango de intensidad):");
        if (ImGui::SliderFloat("##bc", &m_bilateralSigmaColor, 1.0f, 200.0f, "%.0f"))
            m_needsProcess = true;

        ImGui::Text("Sigma Space  (radio espacial):");
        if (ImGui::SliderFloat("##bs", &m_bilateralSigmaSpace, 1.0f, 200.0f, "%.0f"))
            m_needsProcess = true;

        if (m_bilateralD > 15)
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f),
                               "  Diameter > 15 puede ser lento.");

        ImGui::Spacing();
        if (ImGui::Button("Reset##b")) {
            m_bilateralD          = 9;
            m_bilateralSigmaColor = 75.0f;
            m_bilateralSigmaSpace = 75.0f;
            m_needsProcess        = true;
        }
        break;
    }
    }

    // ── Lazy processing ───────────────────────────────────────────────────────
    if (m_needsProcess && !m_original.empty()) {
        reprocess();
        m_needsProcess = false;
    }

    // ── Image row ─────────────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawImageRow();

    // ── Histograms ────────────────────────────────────────────────────────────
    drawHistograms();

    // ── Technical info ────────────────────────────────────────────────────────
    if (!m_original.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Technical Info");
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
        ImGui::Text("Image size    : %d x %d px", m_original.cols, m_original.rows);

        switch (m_mode) {
        case NoiseFilterMode::Gaussian:
            ImGui::Text("Active filter : Gaussian Blur");
            ImGui::Text("Kernel size   : %d x %d", m_gaussKernel, m_gaussKernel);
            ImGui::Text("Sigma         : %s",
                        m_gaussSigma == 0.0f ? "auto" :
                        (std::to_string(m_gaussSigma).substr(0,4) + "f").c_str());
            break;
        case NoiseFilterMode::Median:
            ImGui::Text("Active filter : Median Blur");
            ImGui::Text("Kernel size   : %d x %d", m_medianKernel, m_medianKernel);
            break;
        case NoiseFilterMode::Bilateral:
            ImGui::Text("Active filter : Bilateral Filter");
            ImGui::Text("Diameter      : %d", m_bilateralD);
            ImGui::Text("Sigma Color   : %.0f", m_bilateralSigmaColor);
            ImGui::Text("Sigma Space   : %.0f", m_bilateralSigmaSpace);
            break;
        }

        if (m_lastProcessMs > 0.0f)
            ImGui::Text("Process time  : %.2f ms", m_lastProcessMs);
        ImGui::PopStyleColor();
    }

    // ── Export ────────────────────────────────────────────────────────────────
    if (m_texResult.valid()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Export Result", ImVec2(120, 0))) {
            std::string path = FileDialog::saveImage("noise_result.png");
            if (!path.empty()) {
                bool ok         = cv::imwrite(path, m_result);
                m_exportMsg     = ok ? ("Saved: " + path) : "Error: could not write file.";
                m_exportMsgTime = ImGui::GetTime();
                if (ok)
                    std::fprintf(stdout, "[NoiseReduction] Exported: %s\n", path.c_str());
                else
                    std::fprintf(stderr, "[NoiseReduction] Failed to write: %s\n", path.c_str());
            }
        }

        if (!m_exportMsg.empty() && (ImGui::GetTime() - m_exportMsgTime) < 4.0) {
            ImGui::SameLine();
            bool isError = (m_exportMsg.rfind("Error", 0) == 0);
            ImGui::TextColored(
                isError ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f) : ImVec4(0.4f, 1.0f, 0.6f, 1.0f),
                "%s", m_exportMsg.c_str());
        }
    }

    ImGui::PopID();
}

// ─── Load ─────────────────────────────────────────────────────────────────────

void NoiseReductionModule::loadAndProcess(const std::string& path)
{
    m_original = ImageLoader::load(path);
    if (m_original.empty()) {
        std::fprintf(stderr, "[NoiseReduction] Failed to load: %s\n", path.c_str());
        return;
    }
    m_loadedPath   = path;
    Utils::updateTexture(m_texOrig, m_original);
    m_histOrig     = HistogramUtils::compute(m_original);
    m_statsOrig    = HistogramUtils::computeStats(m_original);
    m_needsProcess = true;
}

// ─── Reprocess ────────────────────────────────────────────────────────────────

void NoiseReductionModule::reprocess()
{
    if (m_original.empty()) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    switch (m_mode) {
    case NoiseFilterMode::Gaussian:  applyGaussian();  break;
    case NoiseFilterMode::Median:    applyMedian();    break;
    case NoiseFilterMode::Bilateral: applyBilateral(); break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    m_lastProcessMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    Utils::updateTexture(m_texResult, m_result);
    m_histResult  = HistogramUtils::compute(m_result);
    m_statsResult = HistogramUtils::computeStats(m_result);
}

// ── Gaussian ──────────────────────────────────────────────────────────────────

void NoiseReductionModule::applyGaussian()
{
    int k = m_gaussKernel;
    if (k < 1) k = 1;
    if (k % 2 == 0) k++;
    // sigma=0: OpenCV computes it automatically as 0.3*(k/2 - 1) + 0.8
    cv::GaussianBlur(m_original, m_result,
                     cv::Size(k, k),
                     static_cast<double>(m_gaussSigma));
}

// ── Median ────────────────────────────────────────────────────────────────────

void NoiseReductionModule::applyMedian()
{
    int k = m_medianKernel;
    if (k < 1) k = 1;
    if (k % 2 == 0) k++;
    cv::medianBlur(m_original, m_result, k);
}

// ── Bilateral ─────────────────────────────────────────────────────────────────

void NoiseReductionModule::applyBilateral()
{
    // bilateralFilter cannot operate in-place; m_result is always a separate Mat
    cv::bilateralFilter(m_original, m_result,
                        std::max(1, m_bilateralD),
                        static_cast<double>(m_bilateralSigmaColor),
                        static_cast<double>(m_bilateralSigmaSpace));
}

// ─── Image row ────────────────────────────────────────────────────────────────

void NoiseReductionModule::drawImageRow()
{
    if (!m_texOrig.valid()) {
        ImGui::TextDisabled("No image loaded. Click Browse... to open an image.");
        return;
    }

    const float avail   = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float imgW    = (avail - spacing) / 2.0f;
    const float aspect  = static_cast<float>(m_texOrig.height) /
                          static_cast<float>(m_texOrig.width);
    const float imgH    = imgW * aspect;

    const char* label =
        m_mode == NoiseFilterMode::Gaussian  ? "Gaussian Blur"    :
        m_mode == NoiseFilterMode::Median    ? "Median Blur"      :
                                               "Bilateral Filter";

    ImGui::Text("Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("%s", label);

    ImGui::Image((ImTextureID)(uintptr_t)m_texOrig.id,   ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texResult.id, ImVec2(imgW, imgH));
}

// ─── Histograms ───────────────────────────────────────────────────────────────

void NoiseReductionModule::drawHistograms()
{
    if (m_original.empty()) return;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Histogramas de intensidad");
    ImGui::TextDisabled("256 bins  |  NORM_MINMAX  |  rango [0, 1]");
    ImGui::TextDisabled("El suavizado reduce picos extremos redistribuyendo intensidades.");
    ImGui::Spacing();

    const float histH   = 100.0f;
    const float availW  = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float histW   = (availW - spacing) / 2.0f;

    auto drawHist = [&](const char* id, const char* label,
                        const std::array<float, 256>& data,
                        const HistogramUtils::ImageStats& stats)
    {
        ImGui::BeginGroup();
        ImGui::TextDisabled("%s", label);
        ImGui::PushID(id);
        ImGui::PlotHistogram("##h", data.data(), 256, 0, nullptr,
                             0.0f, 1.0f, ImVec2(histW, histH));
        ImGui::PopID();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
        ImGui::Text("Media: %.1f   Desv: %.1f", stats.mean, stats.stddev);
        ImGui::Text("Blanco: %.1f%%   Negro: %.1f%%", stats.whitePct, stats.blackPct);
        ImGui::PopStyleColor();
        ImGui::EndGroup();
    };

    const char* resultLabel =
        m_mode == NoiseFilterMode::Gaussian  ? "Gaussian Blur"    :
        m_mode == NoiseFilterMode::Median    ? "Median Blur"      :
                                               "Bilateral Filter";

    drawHist("orig",   "Original",    m_histOrig,   m_statsOrig);
    ImGui::SameLine();
    drawHist("result", resultLabel,   m_histResult, m_statsResult);
}
