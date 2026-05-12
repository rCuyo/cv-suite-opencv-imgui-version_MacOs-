#include "MorphologyModule.h"
#include "core/ImageLoader.h"
#include "core/FileDialog.h"
#include "core/HistogramUtils.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

MorphologyModule::~MorphologyModule()
{
    Utils::deleteTexture(m_texOrig);
    Utils::deleteTexture(m_texResult);
}

// ─── Top-level UI ─────────────────────────────────────────────────────────────

void MorphologyModule::renderUI()
{
    ImGui::PushID("MorphologyModule");

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.2f, 1.0f), "Morphological Operations");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Operation selector ────────────────────────────────────────────────────
    ImGui::Text("Operation:");
    ImGui::SameLine();
    auto opBtn = [&](const char* label, MorphOp op) {
        bool sel = (m_op == op);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::Button(label)) { m_op = op; m_needsProcess = true; }
        if (sel) ImGui::PopStyleColor();
        ImGui::SameLine();
    };
    opBtn("Erosion",  MorphOp::Erosion);
    opBtn("Dilation", MorphOp::Dilation);
    opBtn("Opening",  MorphOp::Opening);
    opBtn("Closing",  MorphOp::Closing);
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

    // ── Operation description ─────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();

    switch (m_op) {
    case MorphOp::Erosion:
        ImGui::TextDisabled("Erosion reduce las regiones blancas eliminando pixeles en los bordes.");
        ImGui::TextDisabled("Reduce regiones y elimina detalles pequenyos.");
        break;
    case MorphOp::Dilation:
        ImGui::TextDisabled("Dilation expande las regiones blancas agregando pixeles en los bordes.");
        ImGui::TextDisabled("Expande regiones y conecta estructuras.");
        break;
    case MorphOp::Opening:
        ImGui::TextDisabled("Opening combina erosion seguida de dilatacion.");
        ImGui::TextDisabled("Util para eliminar ruido manteniendo la estructura principal.");
        break;
    case MorphOp::Closing:
        ImGui::TextDisabled("Closing combina dilatacion seguida de erosion.");
        ImGui::TextDisabled("Util para rellenar huecos y unir regiones cercanas.");
        break;
    }
    ImGui::Spacing();

    // ── Shared parameter controls ─────────────────────────────────────────────
    ImGui::Text("Kernel size (odd):");
    if (ImGui::SliderInt("##ks", &m_kernelSize, 1, 21)) {
        if (m_kernelSize % 2 == 0) m_kernelSize++;
        m_needsProcess = true;
    }

    ImGui::Text("Iterations:");
    if (ImGui::SliderInt("##it", &m_iterations, 1, 10))
        m_needsProcess = true;

    // ── Kernel shape selector ─────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Text("Kernel shape:");
    ImGui::SameLine();
    auto shapeBtn = [&](const char* label, KernelShape shape) {
        bool sel = (m_kernelShape == shape);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::Button(label)) { m_kernelShape = shape; m_needsProcess = true; }
        if (sel) ImGui::PopStyleColor();
        ImGui::SameLine();
    };
    shapeBtn("Rectangle", KernelShape::Rectangle);
    shapeBtn("Ellipse",   KernelShape::Ellipse);
    shapeBtn("Cross",     KernelShape::Cross);
    ImGui::NewLine();
    ImGui::Spacing();

    if (ImGui::Button("Reset")) {
        m_kernelSize   = 3;
        m_iterations   = 1;
        m_kernelShape  = KernelShape::Rectangle;
        m_needsProcess = true;
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

        const char* opName    =
            m_op == MorphOp::Erosion  ? "Erosion"  :
            m_op == MorphOp::Dilation ? "Dilation" :
            m_op == MorphOp::Opening  ? "Opening"  : "Closing";
        const char* shapeName =
            m_kernelShape == KernelShape::Rectangle ? "Rectangle" :
            m_kernelShape == KernelShape::Ellipse   ? "Ellipse"   : "Cross";

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
        ImGui::Text("Image size  : %d x %d px", m_original.cols, m_original.rows);
        ImGui::Text("Operation   : %s", opName);
        ImGui::Text("Kernel size : %d x %d", m_kernelSize, m_kernelSize);
        ImGui::Text("Kernel shape: %s", shapeName);
        ImGui::Text("Iterations  : %d", m_iterations);
        if (m_lastProcessMs > 0.0f)
            ImGui::Text("Process time: %.2f ms", m_lastProcessMs);
        ImGui::PopStyleColor();
    }

    // ── Export ────────────────────────────────────────────────────────────────
    if (m_texResult.valid()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Export Result", ImVec2(120, 0))) {
            std::string path = FileDialog::saveImage("morphology_result.png");
            if (!path.empty()) {
                bool ok         = cv::imwrite(path, m_result);
                m_exportMsg     = ok ? ("Saved: " + path) : "Error: could not write file.";
                m_exportMsgTime = ImGui::GetTime();
                if (ok)
                    std::fprintf(stdout, "[Morphology] Exported: %s\n", path.c_str());
                else
                    std::fprintf(stderr, "[Morphology] Failed to write: %s\n", path.c_str());
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

void MorphologyModule::loadAndProcess(const std::string& path)
{
    m_original = ImageLoader::load(path);
    if (m_original.empty()) {
        std::fprintf(stderr, "[Morphology] Failed to load: %s\n", path.c_str());
        return;
    }
    m_loadedPath   = path;
    Utils::updateTexture(m_texOrig, m_original);
    m_histOrig     = HistogramUtils::compute(m_original);
    m_statsOrig    = HistogramUtils::computeStats(m_original);
    m_needsProcess = true;
}

// ─── Reprocess ────────────────────────────────────────────────────────────────

void MorphologyModule::reprocess()
{
    if (m_original.empty()) return;

    auto t0 = std::chrono::high_resolution_clock::now();
    applyOperation();
    auto t1 = std::chrono::high_resolution_clock::now();
    m_lastProcessMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    Utils::updateTexture(m_texResult, m_result);
    m_histResult  = HistogramUtils::compute(m_result);
    m_statsResult = HistogramUtils::computeStats(m_result);
}

// ─── Apply morphological operation ────────────────────────────────────────────

void MorphologyModule::applyOperation()
{
    int k = m_kernelSize;
    if (k < 1) k = 1;
    if (k % 2 == 0) k++;

    // Build structuring element from selected shape
    const int cvShape =
        m_kernelShape == KernelShape::Ellipse ? cv::MORPH_ELLIPSE :
        m_kernelShape == KernelShape::Cross   ? cv::MORPH_CROSS   :
                                                cv::MORPH_RECT;

    cv::Mat kernel = cv::getStructuringElement(cvShape, cv::Size(k, k));

    switch (m_op) {
    case MorphOp::Erosion:
        cv::erode(m_original, m_result, kernel,
                  cv::Point(-1, -1), m_iterations);
        break;
    case MorphOp::Dilation:
        cv::dilate(m_original, m_result, kernel,
                   cv::Point(-1, -1), m_iterations);
        break;
    case MorphOp::Opening:
        cv::morphologyEx(m_original, m_result, cv::MORPH_OPEN, kernel,
                         cv::Point(-1, -1), m_iterations);
        break;
    case MorphOp::Closing:
        cv::morphologyEx(m_original, m_result, cv::MORPH_CLOSE, kernel,
                         cv::Point(-1, -1), m_iterations);
        break;
    }
}

// ─── Image row ────────────────────────────────────────────────────────────────

void MorphologyModule::drawImageRow()
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
        m_op == MorphOp::Erosion  ? "Erosion"  :
        m_op == MorphOp::Dilation ? "Dilation" :
        m_op == MorphOp::Opening  ? "Opening"  : "Closing";

    ImGui::Text("Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("%s", label);

    ImGui::Image((ImTextureID)(uintptr_t)m_texOrig.id,   ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texResult.id, ImVec2(imgW, imgH));
}

// ─── Histograms ───────────────────────────────────────────────────────────────

void MorphologyModule::drawHistograms()
{
    if (m_original.empty()) return;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Histogramas de intensidad");
    ImGui::TextDisabled("256 bins  |  NORM_MINMAX  |  rango [0, 1]");
    ImGui::TextDisabled("Erosion desplaza el histograma hacia oscuro; Dilation hacia claro.");
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
        m_op == MorphOp::Erosion  ? "Erosion"  :
        m_op == MorphOp::Dilation ? "Dilation" :
        m_op == MorphOp::Opening  ? "Opening"  : "Closing";

    drawHist("orig",   "Original",   m_histOrig,   m_statsOrig);
    ImGui::SameLine();
    drawHist("result", resultLabel,  m_histResult, m_statsResult);
}
