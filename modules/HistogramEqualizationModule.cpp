#include "HistogramEqualizationModule.h"
#include "core/ImageLoader.h"
#include "core/FileDialog.h"
#include "core/HistogramUtils.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

HistogramEqualizationModule::~HistogramEqualizationModule()
{
    Utils::deleteTexture(m_texGray);
    Utils::deleteTexture(m_texEqualized);
}

// ─── Top-level UI ─────────────────────────────────────────────────────────────

void HistogramEqualizationModule::renderUI()
{
    ImGui::PushID("HistEqModule");

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "Histogram Equalization");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Explanation ───────────────────────────────────────────────────────────
    ImGui::TextDisabled("Histogram Equalization mejora el contraste redistribuyendo");
    ImGui::TextDisabled("las intensidades de la imagen de forma uniforme [0, 255].");
    ImGui::TextDisabled("Algoritmo: cv::equalizeHist() sobre imagen en escala de grises.");
    ImGui::Spacing();

    // ── File loader ───────────────────────────────────────────────────────────
    if (ImGui::Button("Browse...", ImVec2(80, 0))) {
        std::string p = FileDialog::openImage();
        if (!p.empty()) loadAndProcess(p);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_loadedPath.empty() ? "No image loaded." : m_loadedPath.c_str());
    ImGui::Spacing();

    // ── Lazy processing ───────────────────────────────────────────────────────
    if (m_needsProcess && !m_gray.empty()) {
        reprocess();
        m_needsProcess = false;
    }

    // ── Image row ─────────────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();
    drawImageRow();

    // ── Histograms ────────────────────────────────────────────────────────────
    drawHistograms();

    // ── Technical info ────────────────────────────────────────────────────────
    if (!m_gray.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Technical Info");
        ImGui::Spacing();

        // Contrast approximation: stddev / 127.5 * 100 (0% = flat, 100% = max bimodal)
        const float contrastOrig = m_statsOrig.stddev    / 127.5f * 100.0f;
        const float contrastEq   = m_statsEqualized.stddev / 127.5f * 100.0f;

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
        ImGui::Text("Image size          : %d x %d px", m_gray.cols, m_gray.rows);
        ImGui::Spacing();
        ImGui::Text("Original");
        ImGui::Text("  Media             : %.1f", m_statsOrig.mean);
        ImGui::Text("  Desv. estándar    : %.1f", m_statsOrig.stddev);
        ImGui::Text("  Contraste aprox.  : %.1f%%", contrastOrig);
        ImGui::Spacing();
        ImGui::Text("Ecualizada");
        ImGui::Text("  Media             : %.1f", m_statsEqualized.mean);
        ImGui::Text("  Desv. estándar    : %.1f", m_statsEqualized.stddev);
        ImGui::Text("  Contraste aprox.  : %.1f%%", contrastEq);
        ImGui::PopStyleColor();
    }

    // ── Export ────────────────────────────────────────────────────────────────
    if (m_texEqualized.valid()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Export Result", ImVec2(120, 0))) {
            std::string path = FileDialog::saveImage("histeq_result.png");
            if (!path.empty()) {
                bool ok         = cv::imwrite(path, m_equalized);
                m_exportMsg     = ok ? ("Saved: " + path) : "Error: could not write file.";
                m_exportMsgTime = ImGui::GetTime();
                if (ok)
                    std::fprintf(stdout, "[HistEq] Exported: %s\n", path.c_str());
                else
                    std::fprintf(stderr, "[HistEq] Failed to write: %s\n", path.c_str());
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

void HistogramEqualizationModule::loadAndProcess(const std::string& path)
{
    m_original = ImageLoader::load(path);
    if (m_original.empty()) {
        std::fprintf(stderr, "[HistEq] Failed to load: %s\n", path.c_str());
        return;
    }
    m_loadedPath = path;
    cv::cvtColor(m_original, m_gray, cv::COLOR_BGR2GRAY);
    Utils::updateTexture(m_texGray, m_gray);
    m_histOrig     = HistogramUtils::compute(m_gray);
    m_statsOrig    = HistogramUtils::computeStats(m_gray);
    m_needsProcess = true;
}

// ─── Reprocess ────────────────────────────────────────────────────────────────

void HistogramEqualizationModule::reprocess()
{
    if (m_gray.empty()) return;

    // Redistribute intensities so the CDF becomes approximately linear
    cv::equalizeHist(m_gray, m_equalized);

    Utils::updateTexture(m_texEqualized, m_equalized);
    m_histEqualized  = HistogramUtils::compute(m_equalized);
    m_statsEqualized = HistogramUtils::computeStats(m_equalized);
}

// ─── Image row ────────────────────────────────────────────────────────────────

void HistogramEqualizationModule::drawImageRow()
{
    if (!m_texGray.valid()) {
        ImGui::TextDisabled("No image loaded. Click Browse... to open an image.");
        return;
    }

    const float avail   = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float imgW    = (avail - spacing) / 2.0f;
    const float aspect  = static_cast<float>(m_texGray.height) /
                          static_cast<float>(m_texGray.width);
    const float imgH    = imgW * aspect;

    ImGui::Text("Original (grayscale)");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("Equalized");

    ImGui::Image((ImTextureID)(uintptr_t)m_texGray.id,      ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texEqualized.id, ImVec2(imgW, imgH));
}

// ─── Histograms ───────────────────────────────────────────────────────────────

void HistogramEqualizationModule::drawHistograms()
{
    if (m_gray.empty()) return;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Histogramas de intensidad");
    ImGui::TextDisabled("256 bins  |  NORM_MINMAX  |  rango [0, 1]");
    ImGui::TextDisabled("La distribucion ecualizada tiende a ser mas uniforme.");
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

    drawHist("orig", "Original",   m_histOrig,      m_statsOrig);
    ImGui::SameLine();
    drawHist("eq",   "Ecualizada", m_histEqualized, m_statsEqualized);
}
