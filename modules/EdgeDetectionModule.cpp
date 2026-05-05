#include "EdgeDetectionModule.h"
#include "core/ImageLoader.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

EdgeDetectionModule::EdgeDetectionModule()
{
    std::memset(m_pathBuf, 0, sizeof(m_pathBuf));
}

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
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Edge Detection");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Mode selector ─────────────────────────────────────────────────────────
    ImGui::Text("Mode:");
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
    ImGui::Text("Image path:");
    ImGui::SetNextItemWidth(-80);
    ImGui::InputText("##path", m_pathBuf, sizeof(m_pathBuf));
    ImGui::SameLine();
    if (ImGui::Button("Load", ImVec2(70, 0)) && m_pathBuf[0] != '\0')
        loadAndProcess(m_pathBuf);
    ImGui::Spacing();

    // ── Canny parameter controls ──────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();

    // ── Gaussian Blur ─────────────────────────────────────────────────────────
    // Before computing gradients, the image is convolved with a Gaussian kernel.
    // This low-pass filter suppresses high-frequency noise so that every steep
    // intensity change we see later corresponds to a real structural edge rather
    // than photon noise or sensor artefacts. A larger kernel blurs more but may
    // merge nearby edges; a smaller kernel preserves fine detail at the cost of
    // more noise in the gradient map.
    ImGui::Text("Gaussian pre-blur kernel (odd, >= 3):");
    if (ImGui::SliderInt("##gk", &m_gaussKernel, 3, 21))
    {
        if (m_gaussKernel % 2 == 0) m_gaussKernel++;
        m_needsProcess = true;
    }

    // ── Sobel Gradient ────────────────────────────────────────────────────────
    // Canny internally runs two 1-D Sobel operators (one horizontal, one
    // vertical) to estimate the first derivative of intensity. The magnitude
    //   G = sqrt(Gx² + Gy²)
    // tells us how sharply intensity changes at each pixel; the direction
    //   θ = atan2(Gy, Gx)
    // tells us which way the edge runs. Aperture size controls the Sobel kernel:
    // 3 is standard (fast, slightly noisier), 5 or 7 give smoother gradients.
    const char* apertures[] = { "3", "5", "7" };
    int apertureIdx = (m_apertureSize - 3) / 2; // 3→0, 5→1, 7→2
    ImGui::Text("Sobel aperture size:");
    if (ImGui::Combo("##ap", &apertureIdx, apertures, 3))
    {
        m_apertureSize = apertureIdx * 2 + 3;
        m_needsProcess = true;
    }

    // ── Hysteresis thresholds ─────────────────────────────────────────────────
    // Non-maximum suppression first thins every gradient "ridge" to a 1-pixel-
    // wide candidate edge by keeping only the local maximum along the gradient
    // direction and zeroing out all other pixels.
    //
    // Hysteresis then classifies the surviving candidates into three groups:
    //   • G > highThresh → definite edge (always kept)
    //   • lowThresh < G <= highThresh → weak edge (kept only if it is 8-connected
    //     to at least one definite edge in the same continuous chain)
    //   • G <= lowThresh → discarded
    //
    // Using two thresholds eliminates isolated weak responses (noise) while
    // preserving edges that are continuous but slightly dim in places.
    ImGui::Text("Canny low threshold:");
    if (ImGui::SliderInt("##cl", &m_cannyLow, 1, m_cannyHigh - 1))
        m_needsProcess = true;

    ImGui::Text("Canny high threshold:");
    if (ImGui::SliderInt("##ch", &m_cannyHigh, m_cannyLow + 1, 500))
        m_needsProcess = true;

    // ── Segmentation-only controls ────────────────────────────────────────────
    if (m_mode == EdgeMode::Segmentation)
    {
        ImGui::Spacing();
        ImGui::Text("Min contour area (px²):");
        if (ImGui::SliderInt("##mca", &m_minContourArea, 0, 5000))
            m_needsProcess = true;
    }

    // ── Lazy processing ───────────────────────────────────────────────────────
    if (m_needsProcess && !m_gray.empty()) {
        reprocess();
        m_needsProcess = false;
    }

    // ── TODO: Histogram visualisation ─────────────────────────────────────────
    // TODO: draw a gradient-magnitude histogram using ImDrawList so users can
    // pick Canny thresholds based on the actual distribution in their image.

    // ── TODO: ML comparison ───────────────────────────────────────────────────
    // TODO: run a learned edge detector (HED or DEXINED via ONNX Runtime) on the
    // same image and place it next to the Canny output for a side-by-side
    // quality comparison at matched runtime budgets.

    // ── Image viewer ──────────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawImageRow();

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
    // Step 1 — Gaussian blur (noise suppression, see comment in renderUI)
    cv::Mat blurred;
    int ksize = m_gaussKernel;
    if (ksize < 3)       ksize = 3;
    if (ksize % 2 == 0)  ksize++;
    cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), 0);

    // Step 2-4 — Sobel gradient → NMS → Hysteresis (all inside cv::Canny)
    cv::Canny(blurred, m_edges, m_cannyLow, m_cannyHigh, m_apertureSize);

    Utils::updateTexture(m_texEdges, m_edges);
    Utils::deleteTexture(m_texContours); // not displayed in this mode
}

// ── Segmentation ──────────────────────────────────────────────────────────────

void EdgeDetectionModule::processSegmentation(const cv::Mat& gray)
{
    // Gaussian blur + Canny (same pipeline as OCR mode)
    cv::Mat blurred;
    int ksize = m_gaussKernel;
    if (ksize < 3)       ksize = 3;
    if (ksize % 2 == 0)  ksize++;
    cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), 0);
    cv::Canny(blurred, m_edges, m_cannyLow, m_cannyHigh, m_apertureSize);

    Utils::updateTexture(m_texEdges, m_edges);

    // Find external contours from the closed edge chains.
    // RETR_EXTERNAL: only outermost contours (faster, avoids nested shapes).
    // CHAIN_APPROX_SIMPLE: compress horizontal/vertical/diagonal segments to
    //   their endpoints only, saving memory and drawing time.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(m_edges.clone(), contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw contours on a colour copy of the original
    cv::cvtColor(gray, m_contourViz, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < static_cast<int>(contours.size()); ++i)
    {
        if (cv::contourArea(contours[i]) < m_minContourArea) continue;

        // Random per-contour colour so overlapping objects are distinguishable
        cv::Scalar colour(
            (i * 47 + 80)  % 200 + 55,
            (i * 113 + 40) % 200 + 55,
            (i * 211 + 60) % 200 + 55);

        cv::drawContours(m_contourViz, contours, i, colour, 2);

        // Label each contour with its area
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
        ImGui::TextDisabled("No image loaded. Enter a path above and click Load.");
        return;
    }

    bool hasContours = m_texContours.valid();
    int  cols        = hasContours ? 3 : 2;

    float avail   = ImGui::GetContentRegionAvail().x;
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float imgW    = (avail - spacing * (cols - 1)) / static_cast<float>(cols);
    float aspect  = static_cast<float>(m_texOrig.height) / static_cast<float>(m_texOrig.width);
    float imgH    = imgW * aspect;

    // ── Labels ────────────────────────────────────────────────────────────────
    ImGui::Text("Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("Canny Edges");
    if (hasContours) {
        ImGui::SameLine(2.0f * (imgW + spacing));
        ImGui::Text("Contours");
    }

    // ── Images ────────────────────────────────────────────────────────────────
    ImGui::Image((ImTextureID)(uintptr_t)m_texOrig.id,   ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texEdges.id,  ImVec2(imgW, imgH));
    if (hasContours) {
        ImGui::SameLine();
        ImGui::Image((ImTextureID)(uintptr_t)m_texContours.id, ImVec2(imgW, imgH));
    }
}
