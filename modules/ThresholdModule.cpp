#include "ThresholdModule.h"
#include "core/ImageLoader.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

ThresholdModule::ThresholdModule()
{
    std::memset(m_pathBuf, 0, sizeof(m_pathBuf));
}

ThresholdModule::~ThresholdModule()
{
    Utils::deleteTexture(m_texOrig);
    Utils::deleteTexture(m_texA);
    Utils::deleteTexture(m_texB);
}

// ─── Top-level UI ─────────────────────────────────────────────────────────────

void ThresholdModule::renderUI()
{
    ImGui::PushID("ThresholdModule");

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Thresholding");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Mode selector ─────────────────────────────────────────────────────────
    ImGui::Text("Mode:");
    ImGui::SameLine();
    auto modeBtn = [&](const char* label, ThresholdMode mode) {
        bool sel = (m_mode == mode);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::Button(label)) { m_mode = mode; m_needsProcess = true; }
        if (sel) ImGui::PopStyleColor();
        ImGui::SameLine();
    };
    modeBtn("OCR",        ThresholdMode::OCR);
    modeBtn("Medical",    ThresholdMode::MedicalSegmentation);
    modeBtn("Industrial", ThresholdMode::IndustrialInspection);
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

    // ── Mode-specific parameter controls ─────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();

    switch (m_mode)
    {
    // ── OCR ──────────────────────────────────────────────────────────────────
    case ThresholdMode::OCR:
    {
        ImGui::TextDisabled("Otsu finds a global threshold by maximising inter-class variance.");
        ImGui::TextDisabled("Adaptive computes a local threshold per pixel neighbourhood.");
        ImGui::Spacing();

        ImGui::Text("Adaptive block size (odd):");
        if (ImGui::SliderInt("##blk", &m_adaptiveBlockSize, 3, 51))
        {
            if (m_adaptiveBlockSize % 2 == 0) m_adaptiveBlockSize++; // keep odd
            m_needsProcess = true;
        }
        ImGui::Text("Adaptive C constant:");
        if (ImGui::SliderInt("##C", &m_adaptiveC, -10, 20))
            m_needsProcess = true;
        break;
    }

    // ── Medical ───────────────────────────────────────────────────────────────
    case ThresholdMode::MedicalSegmentation:
    {
        ImGui::TextDisabled("Two-level thresholding isolates a tissue intensity band.");
        ImGui::Spacing();
        ImGui::Text("Low threshold:");
        if (ImGui::SliderInt("##ml", &m_medLow, 0, 254))
        {
            m_medLow = std::min(m_medLow, m_medHigh - 1);
            m_needsProcess = true;
        }
        ImGui::Text("High threshold:");
        if (ImGui::SliderInt("##mh", &m_medHigh, 1, 255))
        {
            m_medHigh = std::max(m_medHigh, m_medLow + 1);
            m_needsProcess = true;
        }
        break;
    }

    // ── Industrial ───────────────────────────────────────────────────────────
    case ThresholdMode::IndustrialInspection:
    {
        // Why thresholding over ML in industrial settings?
        //
        // 1. Determinism — thresholding is a pure mathematical function; the
        //    same input always produces the same output. Safety-critical
        //    manufacturing requires this kind of auditability.
        //
        // 2. No training data — defect samples are rare by design in
        //    well-run factories. Deep learning needs thousands of labelled
        //    examples; thresholding needs a single intensity value.
        //
        // 3. Speed — a global threshold runs in O(N) per frame, fitting easily
        //    in a real-time pipeline (>100 fps on CPU). CNN inference is slower.
        //
        // 4. Controlled illumination — factory floors use structured, constant
        //    lighting. Under these conditions intensity values are stable and
        //    a fixed threshold is highly reliable.
        //
        // 5. Explainability — QA engineers can immediately understand and tune
        //    the rule. An ML model is a black box.
        //
        ImGui::TextDisabled("Threshold-based defect detection.");
        ImGui::TextDisabled("(Preferred over ML: deterministic, fast, no training data)");
        ImGui::Spacing();

        ImGui::Text("Defect darkness threshold:");
        if (ImGui::SliderInt("##dt", &m_defectThresh, 1, 120))
            m_needsProcess = true;

        ImGui::Text("Pre-blur kernel size:");
        if (ImGui::SliderInt("##gk", &m_gaussBlurSize, 1, 21))
        {
            if (m_gaussBlurSize % 2 == 0) m_gaussBlurSize++;
            m_needsProcess = true;
        }
        break;
    }
    }

    // ── Lazy processing ───────────────────────────────────────────────────────
    if (m_needsProcess && !m_gray.empty()) {
        reprocess();
        m_needsProcess = false;
    }

    // ── TODO: Histogram visualisation ─────────────────────────────────────────
    // TODO: draw a real-time intensity histogram of m_gray using ImDrawList
    // to help users choose threshold values visually.

    // ── TODO: ML comparison ───────────────────────────────────────────────────
    // TODO: run a lightweight U-Net or SegFormer inference on the same image
    // and display the output alongside the threshold result so users can
    // compare precision, recall, and runtime.

    // ── Image viewer ──────────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawImageRow();

    ImGui::PopID();
}

// ─── Load & process ───────────────────────────────────────────────────────────

void ThresholdModule::loadAndProcess(const std::string& path)
{
    m_original = ImageLoader::load(path);
    if (m_original.empty()) {
        std::fprintf(stderr, "[Threshold] Failed to load: %s\n", path.c_str());
        return;
    }

    cv::cvtColor(m_original, m_gray, cv::COLOR_BGR2GRAY);

    Utils::updateTexture(m_texOrig, m_original);
    m_needsProcess = true;
}

// ─── Processing pipeline ──────────────────────────────────────────────────────

void ThresholdModule::reprocess()
{
    if (m_gray.empty()) return;

    switch (m_mode)
    {
    case ThresholdMode::OCR:                 processOCR(m_gray);         break;
    case ThresholdMode::MedicalSegmentation: processMedical(m_gray);     break;
    case ThresholdMode::IndustrialInspection:processIndustrial(m_gray);  break;
    }
}

// ── OCR ───────────────────────────────────────────────────────────────────────

void ThresholdModule::processOCR(const cv::Mat& gray)
{
    m_labelA = "Otsu";
    m_labelB = "Adaptive";

    // Otsu: automatically picks the threshold that maximises inter-class
    // variance between foreground (text) and background pixels.
    cv::threshold(gray, m_resultA, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Adaptive: computes a local mean over a neighbourhood and subtracts C.
    // Handles uneven illumination (e.g. a curved page) where a single global
    // threshold would misclassify large regions.
    int blockSize = m_adaptiveBlockSize;
    if (blockSize < 3)        blockSize = 3;
    if (blockSize % 2 == 0)   blockSize++;

    cv::adaptiveThreshold(gray, m_resultB, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY,
                          blockSize,
                          static_cast<double>(m_adaptiveC));

    Utils::updateTexture(m_texA, m_resultA);
    Utils::updateTexture(m_texB, m_resultB);
}

// ── Medical ───────────────────────────────────────────────────────────────────

void ThresholdModule::processMedical(const cv::Mat& gray)
{
    m_labelA = "Segmented Region";
    m_labelB = "";

    // Pixels with intensity in [medLow, medHigh] are considered the region of
    // interest (e.g. a specific tissue type in an ultrasound or CT slice).
    cv::Mat lower, upper;
    cv::threshold(gray, lower, m_medLow,  255, cv::THRESH_BINARY);
    cv::threshold(gray, upper, m_medHigh, 255, cv::THRESH_BINARY_INV);

    cv::Mat mask;
    cv::bitwise_and(lower, upper, mask);

    // Create a colour overlay: original in grayscale, ROI highlighted in red
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    bgr.setTo(cv::Scalar(0, 0, 220), mask);

    m_resultA = bgr;
    Utils::updateTexture(m_texA, m_resultA);
    Utils::deleteTexture(m_texB); // not used in this mode
}

// ── Industrial ────────────────────────────────────────────────────────────────

void ThresholdModule::processIndustrial(const cv::Mat& gray)
{
    m_labelA = "Defect Map";
    m_labelB = "";

    // Pre-blur: remove high-frequency sensor noise so that pixel-level texture
    // variation does not produce spurious defect detections.
    cv::Mat blurred;
    int ksize = m_gaussBlurSize;
    if (ksize % 2 == 0)  ksize++;
    if (ksize < 1)       ksize = 1;
    cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), 0);

    // Dark-defect detection: cracks, voids, and missing material appear
    // significantly darker than the surrounding surface under structured light.
    cv::Mat defectMask;
    cv::threshold(blurred, defectMask, m_defectThresh, 255, cv::THRESH_BINARY_INV);

    // Morphological cleanup: remove single-pixel noise from the defect mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(defectMask, defectMask, cv::MORPH_OPEN, kernel);

    // Visualise: green overlay on the defect regions
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    bgr.setTo(cv::Scalar(0, 220, 0), defectMask);

    // Draw bounding boxes around connected defect components
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(defectMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : contours) {
        if (cv::contourArea(c) < 10) continue; // ignore micro-noise
        cv::Rect bbox = cv::boundingRect(c);
        cv::rectangle(bgr, bbox, cv::Scalar(0, 60, 255), 1);
    }

    m_resultA = bgr;
    Utils::updateTexture(m_texA, m_resultA);
    Utils::deleteTexture(m_texB);
}

// ─── Image row renderer ───────────────────────────────────────────────────────

void ThresholdModule::drawImageRow()
{
    if (!m_texOrig.valid()) {
        ImGui::TextDisabled("No image loaded. Enter a path above and click Load.");
        return;
    }

    // How many images are we displaying?
    bool hasB = m_texB.valid();
    int  cols = hasB ? 3 : 2;

    float avail   = ImGui::GetContentRegionAvail().x;
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float imgW    = (avail - spacing * (cols - 1)) / static_cast<float>(cols);

    // Maintain aspect ratio from the original
    float aspect  = static_cast<float>(m_texOrig.height) / static_cast<float>(m_texOrig.width);
    float imgH    = imgW * aspect;

    // ── Labels row ────────────────────────────────────────────────────────────
    ImGui::Text("Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("%s", m_labelA.c_str());
    if (hasB) {
        ImGui::SameLine(2.0f * (imgW + spacing));
        ImGui::Text("%s", m_labelB.c_str());
    }

    // ── Images row ────────────────────────────────────────────────────────────
    ImGui::Image((ImTextureID)(uintptr_t)m_texOrig.id, ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texA.id, ImVec2(imgW, imgH));
    if (hasB) {
        ImGui::SameLine();
        ImGui::Image((ImTextureID)(uintptr_t)m_texB.id, ImVec2(imgW, imgH));
    }
}
