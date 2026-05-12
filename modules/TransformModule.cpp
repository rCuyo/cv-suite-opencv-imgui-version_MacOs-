#include "TransformModule.h"
#include "core/ImageLoader.h"
#include "core/FileDialog.h"
#include "core/HistogramUtils.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

TransformModule::~TransformModule()
{
    Utils::deleteTexture(m_texOrig);
    Utils::deleteTexture(m_texResult);
}

// ─── Top-level UI ─────────────────────────────────────────────────────────────

void TransformModule::renderUI()
{
    ImGui::PushID("TransformModule");

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.5f, 1.0f), "Geometric Transformations");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Mode selector — same button-highlight pattern as other modules ─────────
    ImGui::Text("Mode:");
    ImGui::SameLine();
    auto modeBtn = [&](const char* label, TransformMode mode) {
        bool sel = (m_mode == mode);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::Button(label)) { m_mode = mode; m_needsProcess = true; }
        if (sel) ImGui::PopStyleColor();
        ImGui::SameLine();
    };
    modeBtn("Translation", TransformMode::Translation);
    modeBtn("Rotation",    TransformMode::Rotation);
    modeBtn("Scaling",     TransformMode::Scaling);
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
    // ── Translation ───────────────────────────────────────────────────────────
    case TransformMode::Translation:
    {
        ImGui::TextDisabled("Translation desplaza la imagen en los ejes X e Y sin modificar su orientacion.");
        ImGui::Spacing();
        ImGui::Text("Click and drag on the result image to translate interactively.");
        ImGui::Spacing();
        if (ImGui::Button("Reset Translation")) {
            m_translationX = 0.0f;
            m_translationY = 0.0f;
            m_needsProcess = true;
        }
        break;
    }

    // ── Rotation ──────────────────────────────────────────────────────────────
    case TransformMode::Rotation:
    {
        ImGui::TextDisabled("Rotation gira la imagen alrededor de un punto, normalmente el centro.");
        ImGui::Spacing();
        ImGui::Text("Rotation angle:");
        if (ImGui::SliderFloat("##angle", &m_angle, -180.0f, 180.0f, "%.1f deg"))
            m_needsProcess = true;
        ImGui::Spacing();
        if (ImGui::Button("Reset Rotation")) {
            m_angle        = 0.0f;
            m_needsProcess = true;
        }
        break;
    }

    // ── Scaling ───────────────────────────────────────────────────────────────
    case TransformMode::Scaling:
    {
        ImGui::TextDisabled("Scaling modifica el tamanyo de la imagen utilizando");
        ImGui::TextDisabled("factores de escala horizontales y verticales.");
        ImGui::TextDisabled("Scale > 1: amplia imagen.  Scale < 1: reduce imagen.");
        ImGui::Spacing();

        if (ImGui::Checkbox("Uniform Scale  (X = Y)", &m_uniformScale)) {
            if (m_uniformScale) m_scaleY = m_scaleX;
            m_needsProcess = true;
        }

        if (m_uniformScale) {
            ImGui::Text("Scale (uniform):");
            if (ImGui::SliderFloat("##su", &m_scaleX, 0.1f, 3.0f, "%.2fx")) {
                m_scaleY       = m_scaleX;
                m_needsProcess = true;
            }
        } else {
            ImGui::SameLine();
            if (ImGui::Checkbox("Lock Aspect Ratio", &m_lockAspect))
                m_needsProcess = true;

            const float aspect = m_original.empty() ? 1.0f
                : static_cast<float>(m_original.rows) / static_cast<float>(m_original.cols);

            ImGui::Text("Scale X:");
            if (ImGui::SliderFloat("##sx", &m_scaleX, 0.1f, 3.0f, "%.2fx")) {
                if (m_lockAspect) m_scaleY = m_scaleX * aspect;
                m_needsProcess = true;
            }
            ImGui::Text("Scale Y:");
            if (ImGui::SliderFloat("##sy", &m_scaleY, 0.1f, 3.0f, "%.2fx")) {
                if (m_lockAspect) m_scaleX = m_scaleY / aspect;
                m_needsProcess = true;
            }
        }

        ImGui::Spacing();
        if (ImGui::Button("Reset Scaling")) {
            m_scaleX       = 1.0f;
            m_scaleY       = 1.0f;
            m_needsProcess = true;
        }
        break;
    }
    }

    // ── Lazy processing — runs once per change, not every frame ───────────────
    if (m_needsProcess && !m_original.empty()) {
        reprocess();
        m_needsProcess = false;
    }

    // ── Image row (drag detection happens here) ────────────────────────────────
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
        switch (m_mode)
        {
        case TransformMode::Translation:
            ImGui::Text("Original     : %d x %d px", m_original.cols, m_original.rows);
            ImGui::Text("TranslationX : %.1f px", m_translationX);
            ImGui::Text("TranslationY : %.1f px", m_translationY);
            break;
        case TransformMode::Rotation:
            ImGui::Text("Original     : %d x %d px", m_original.cols, m_original.rows);
            ImGui::Text("Angle        : %.1f deg", m_angle);
            break;
        case TransformMode::Scaling:
            ImGui::Text("Original     : %d x %d px", m_original.cols, m_original.rows);
            if (!m_result.empty())
                ImGui::Text("Scaled       : %d x %d px", m_result.cols, m_result.rows);
            ImGui::Text("Scale X      : %.2fx", m_scaleX);
            ImGui::Text("Scale Y      : %.2fx", m_scaleY);
            break;
        }
        ImGui::PopStyleColor();
    }

    // ── Export ────────────────────────────────────────────────────────────────
    if (m_texResult.valid()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Export Result", ImVec2(120, 0))) {
            std::string path = FileDialog::saveImage("transform_result.png");
            if (!path.empty()) {
                bool ok         = cv::imwrite(path, m_result);
                m_exportMsg     = ok ? ("Saved: " + path) : "Error: could not write file.";
                m_exportMsgTime = ImGui::GetTime();
                if (ok)
                    std::fprintf(stdout, "[TransformModule] Exported: %s\n", path.c_str());
                else
                    std::fprintf(stderr, "[TransformModule] Failed to write: %s\n", path.c_str());
            }
        }

        // Show success/error for 4 seconds after export
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

// ─── Load & process ───────────────────────────────────────────────────────────

void TransformModule::loadAndProcess(const std::string& path)
{
    m_original = ImageLoader::load(path);
    if (m_original.empty()) {
        std::fprintf(stderr, "[TransformModule] Failed to load: %s\n", path.c_str());
        return;
    }
    m_loadedPath   = path;
    m_translationX = 0.0f;
    m_translationY = 0.0f;
    m_angle        = 0.0f;
    m_scaleX       = 1.0f;
    m_scaleY       = 1.0f;
    Utils::updateTexture(m_texOrig, m_original);
    m_histOrig     = HistogramUtils::compute(m_original);
    m_statsOrig    = HistogramUtils::computeStats(m_original);
    m_needsProcess = true;
}

// ─── Reprocess ────────────────────────────────────────────────────────────────

void TransformModule::reprocess()
{
    if (m_original.empty()) return;

    switch (m_mode) {
    case TransformMode::Translation: applyTranslation(); break;
    case TransformMode::Rotation:    applyRotation();    break;
    case TransformMode::Scaling:     applyScaling();     break;
    }

    Utils::updateTexture(m_texResult, m_result);
    m_histResult  = HistogramUtils::compute(m_result);
    m_statsResult = HistogramUtils::computeStats(m_result);
}

// ── Translation ───────────────────────────────────────────────────────────────

void TransformModule::applyTranslation()
{
    // Build 2×3 affine translation matrix: T = [[1, 0, tx], [0, 1, ty]]
    cv::Mat M = (cv::Mat_<double>(2, 3) <<
        1.0, 0.0, static_cast<double>(m_translationX),
        0.0, 1.0, static_cast<double>(m_translationY));

    cv::warpAffine(m_original, m_result, M, m_original.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

// ── Rotation ──────────────────────────────────────────────────────────────────

void TransformModule::applyRotation()
{
    // Rotation around image center; scale=1.0 preserves original canvas size
    cv::Point2f center(
        static_cast<float>(m_original.cols) * 0.5f,
        static_cast<float>(m_original.rows) * 0.5f);

    cv::Mat M = cv::getRotationMatrix2D(center, static_cast<double>(m_angle), 1.0);
    cv::warpAffine(m_original, m_result, M, m_original.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

// ── Scaling ───────────────────────────────────────────────────────────────────

void TransformModule::applyScaling()
{
    // Clamp to avoid zero or negative dimensions (OpenCV assertion guard)
    const float sx = std::max(0.01f, m_scaleX);
    const float sy = std::max(0.01f, m_scaleY);

    const int newW = std::max(1, static_cast<int>(std::round(m_original.cols * sx)));
    const int newH = std::max(1, static_cast<int>(std::round(m_original.rows * sy)));

    // INTER_AREA is better for downscaling; INTER_LINEAR for upscaling
    const int interp = (sx < 1.0f || sy < 1.0f) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(m_original, m_result, cv::Size(newW, newH), 0, 0, interp);
}

// ─── Image row renderer ───────────────────────────────────────────────────────

void TransformModule::drawImageRow()
{
    if (!m_texOrig.valid()) {
        ImGui::TextDisabled("No image loaded. Click Browse... to open an image.");
        return;
    }

    const float avail   = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float imgW    = (avail - spacing) / 2.0f;

    // Each panel uses its own texture's aspect so scaling distortion is visible
    const float origH   = imgW * static_cast<float>(m_texOrig.height)   / static_cast<float>(m_texOrig.width);
    const float resultH = imgW * static_cast<float>(m_texResult.height) / static_cast<float>(m_texResult.width);

    const char* resultLabel =
        m_mode == TransformMode::Translation ? "Translated" :
        m_mode == TransformMode::Rotation    ? "Rotated"    : "Scaled";

    ImGui::Text("Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("%s", resultLabel);

    ImGui::Image((ImTextureID)(uintptr_t)m_texOrig.id,   ImVec2(imgW, origH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texResult.id, ImVec2(imgW, resultH));

    // ── Interactive drag — Translation mode only ───────────────────────────────
    // IsItemHovered/IsMouseClicked query the last rendered item (result image).
    // Screen-to-image scaling ensures 1:1 visual tracking of the mouse.
    if (m_mode == TransformMode::Translation)
    {
        const float scaleX = static_cast<float>(m_original.cols) / imgW;
        const float scaleY = static_cast<float>(m_original.rows) / origH;

        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            m_isDragging      = true;
            ImVec2 mp         = ImGui::GetMousePos();
            m_dragStartMouseX = mp.x;
            m_dragStartMouseY = mp.y;
            m_dragStartTX     = m_translationX;
            m_dragStartTY     = m_translationY;
        }

        if (m_isDragging) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                ImVec2      mp    = ImGui::GetMousePos();
                const float newTX = m_dragStartTX + (mp.x - m_dragStartMouseX) * scaleX;
                const float newTY = m_dragStartTY + (mp.y - m_dragStartMouseY) * scaleY;
                if (newTX != m_translationX || newTY != m_translationY) {
                    m_translationX = newTX;
                    m_translationY = newTY;
                    m_needsProcess = true;
                }
            } else {
                m_isDragging = false;
            }
        }
    }
}

// ─── Histogram renderer ───────────────────────────────────────────────────────

void TransformModule::drawHistograms()
{
    if (m_original.empty()) return;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Intensity histograms");
    ImGui::TextDisabled("256 bins  |  NORM_MINMAX  |  range [0, 1]");
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
        ImGui::Text("Mean: %.1f   Std: %.1f", stats.mean, stats.stddev);
        ImGui::Text("White: %.1f%%   Black: %.1f%%", stats.whitePct, stats.blackPct);
        ImGui::PopStyleColor();
        ImGui::EndGroup();
    };

    const char* resultLabel =
        m_mode == TransformMode::Translation ? "Translated" :
        m_mode == TransformMode::Rotation    ? "Rotated"    : "Scaled";
    drawHist("orig",   "Original",  m_histOrig,   m_statsOrig);
    ImGui::SameLine();
    drawHist("result", resultLabel, m_histResult, m_statsResult);
}
