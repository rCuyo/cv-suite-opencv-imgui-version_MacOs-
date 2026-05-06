#include "ThresholdModule.h"
#include "core/ImageLoader.h"
#include "core/FileDialog.h"
#include "core/HistogramUtils.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

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
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Umbralización");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Mode selector ─────────────────────────────────────────────────────────
    ImGui::Text("Modo:");
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
    if (ImGui::Button("Explorar...", ImVec2(90, 0))) {
        std::string p = FileDialog::openImage();
        if (!p.empty()) loadAndProcess(p);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_loadedPath.empty() ? "Sin imagen cargada" : m_loadedPath.c_str());
    ImGui::Spacing();

    // ── Mode-specific parameter controls ─────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();

    switch (m_mode)
    {
    // ── OCR ──────────────────────────────────────────────────────────────────
    case ThresholdMode::OCR:
    {
        ImGui::TextDisabled("Otsu calcula un umbral global maximizando la varianza entre clases.");
        ImGui::TextDisabled("Adaptativo calcula un umbral local por vecindad de píxel.");
        ImGui::Spacing();
        ImGui::Text("Tamaño de bloque adaptativo (impar):");
        if (ImGui::SliderInt("##blk", &m_adaptiveBlockSize, 3, 51))
        {
            if (m_adaptiveBlockSize % 2 == 0) m_adaptiveBlockSize++;
            m_needsProcess = true;
        }
        ImGui::Text("Constante C adaptativa:");
        if (ImGui::SliderInt("##C", &m_adaptiveC, -10, 20))
            m_needsProcess = true;
        break;
    }

    // ── Medical ───────────────────────────────────────────────────────────────
    case ThresholdMode::MedicalSegmentation:
    {
        ImGui::TextDisabled("Umbralización de dos niveles aísla una banda de intensidad de tejido.");
        ImGui::Spacing();
        ImGui::Text("Umbral inferior:");
        if (ImGui::SliderInt("##ml", &m_medLow, 0, 254))
        {
            m_medLow = std::min(m_medLow, m_medHigh - 1);
            m_needsProcess = true;
        }
        ImGui::Text("Umbral superior:");
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
        // 1. Determinism — same input always produces same output; required for
        //    safety-critical manufacturing audits.
        // 2. No training data — defect samples are rare; DL needs thousands of
        //    labelled examples, thresholding needs one intensity value.
        // 3. Speed — O(N) per frame, fits real-time pipelines (>100 fps on CPU).
        // 4. Controlled illumination — factory structured lighting makes intensity
        //    values stable and a fixed threshold highly reliable.
        // 5. Explainability — QA engineers can immediately understand and tune
        //    the rule; an ML model is a black box.
        //
        ImGui::TextDisabled("Detección de defectos por umbral de intensidad.");
        ImGui::TextDisabled("(Preferido sobre ML: determinista, rápido, sin datos de entrenamiento)");
        ImGui::Spacing();
        ImGui::Text("Umbral de oscuridad del defecto:");
        if (ImGui::SliderInt("##dt", &m_defectThresh, 1, 120))
            m_needsProcess = true;
        ImGui::Text("Tamaño de kernel de desenfoque:");
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

    // ── Image viewer ──────────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawImageRow();

    // ── Histograms ────────────────────────────────────────────────────────────
    drawHistograms();

    // ── Export ────────────────────────────────────────────────────────────────
    if (m_texA.valid()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        if (ImGui::Button("Exportar resultado", ImVec2(140, 0))) {
            std::string p = FileDialog::saveImage();
            if (!p.empty())
                cv::imwrite(p, m_resultA);
        }
    }

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
    m_loadedPath = path;
    cv::cvtColor(m_original, m_gray, cv::COLOR_BGR2GRAY);
    Utils::updateTexture(m_texOrig, m_original);
    m_histOrig  = HistogramUtils::compute(m_gray);
    m_statsOrig = HistogramUtils::computeStats(m_gray);
    m_needsProcess = true;
}

// ─── Processing pipeline ──────────────────────────────────────────────────────

void ThresholdModule::reprocess()
{
    if (m_gray.empty()) return;

    switch (m_mode)
    {
    case ThresholdMode::OCR:                  processOCR(m_gray);          break;
    case ThresholdMode::MedicalSegmentation:  processMedical(m_gray);      break;
    case ThresholdMode::IndustrialInspection: processIndustrial(m_gray);   break;
    }

    m_histA  = HistogramUtils::compute(m_resultA);
    m_statsA = HistogramUtils::computeStats(m_resultA);
    if (!m_resultB.empty()) {
        m_histB  = HistogramUtils::compute(m_resultB);
        m_statsB = HistogramUtils::computeStats(m_resultB);
    } else {
        m_histB.fill(0.0f);
        m_statsB = {};
    }
}

// ── OCR ───────────────────────────────────────────────────────────────────────

void ThresholdModule::processOCR(const cv::Mat& gray)
{
    m_labelA = "Otsu";
    m_labelB = "Adaptive";

    cv::threshold(gray, m_resultA, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    int blockSize = m_adaptiveBlockSize;
    if (blockSize < 3)      blockSize = 3;
    if (blockSize % 2 == 0) blockSize++;

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
    m_labelA = "Región segmentada";
    m_labelB = "";

    cv::Mat lower, upper;
    cv::threshold(gray, lower, m_medLow,  255, cv::THRESH_BINARY);
    cv::threshold(gray, upper, m_medHigh, 255, cv::THRESH_BINARY_INV);

    cv::Mat mask;
    cv::bitwise_and(lower, upper, mask);

    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    bgr.setTo(cv::Scalar(0, 0, 220), mask);

    m_resultA = bgr;
    m_resultB = cv::Mat();
    Utils::updateTexture(m_texA, m_resultA);
    Utils::deleteTexture(m_texB);
}

// ── Industrial ────────────────────────────────────────────────────────────────

void ThresholdModule::processIndustrial(const cv::Mat& gray)
{
    m_labelA = "Mapa de defectos";
    m_labelB = "";

    cv::Mat blurred;
    int ksize = m_gaussBlurSize;
    if (ksize % 2 == 0) ksize++;
    if (ksize < 1)      ksize = 1;
    cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), 0);

    cv::Mat defectMask;
    cv::threshold(blurred, defectMask, m_defectThresh, 255, cv::THRESH_BINARY_INV);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(defectMask, defectMask, cv::MORPH_OPEN, kernel);

    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    bgr.setTo(cv::Scalar(0, 220, 0), defectMask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(defectMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : contours) {
        if (cv::contourArea(c) < 10) continue;
        cv::Rect bbox = cv::boundingRect(c);
        cv::rectangle(bgr, bbox, cv::Scalar(0, 60, 255), 1);
    }

    m_resultA = bgr;
    m_resultB = cv::Mat();
    Utils::updateTexture(m_texA, m_resultA);
    Utils::deleteTexture(m_texB);
}

// ─── Image row renderer ───────────────────────────────────────────────────────

void ThresholdModule::drawImageRow()
{
    if (!m_texOrig.valid()) {
        ImGui::TextDisabled("Sin imagen cargada. Haz clic en Explorar para abrir una imagen.");
        return;
    }

    bool hasB = m_texB.valid();
    int  cols = hasB ? 3 : 2;

    float avail   = ImGui::GetContentRegionAvail().x;
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float imgW    = (avail - spacing * (cols - 1)) / static_cast<float>(cols);
    float aspect  = static_cast<float>(m_texOrig.height) / static_cast<float>(m_texOrig.width);
    float imgH    = imgW * aspect;

    ImGui::Text("Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("%s", m_labelA.c_str());
    if (hasB) {
        ImGui::SameLine(2.0f * (imgW + spacing));
        ImGui::Text("%s", m_labelB.c_str());
    }

    ImGui::Image((ImTextureID)(uintptr_t)m_texOrig.id, ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texA.id,    ImVec2(imgW, imgH));
    if (hasB) {
        ImGui::SameLine();
        ImGui::Image((ImTextureID)(uintptr_t)m_texB.id, ImVec2(imgW, imgH));
    }
}

// ─── Histogram renderer ───────────────────────────────────────────────────────

void ThresholdModule::drawHistograms()
{
    if (m_gray.empty()) return;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Histogramas de intensidad");
    ImGui::TextDisabled("256 bandas  |  NORM_MINMAX  |  rango [0, 1]");
    ImGui::Spacing();

    const float histH   = 100.0f;
    const float availW  = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const bool  hasB    = (m_mode == ThresholdMode::OCR && !m_resultB.empty());
    const int   cols    = hasB ? 3 : 2;
    const float histW   = (availW - spacing * (cols - 1)) / static_cast<float>(cols);

    auto drawHist = [&](const char* id, const char* label,
                        const std::array<float, 256>& data,
                        const HistogramUtils::ImageStats& stats)
    {
        ImGui::BeginGroup();

        // Column header
        ImGui::TextDisabled("%s", label);

        // Histogram plot — explicit scale [0, 1] prevents ImGui auto-scaling
        // from collapsing binary peaks into flat lines
        ImGui::PushID(id);
        ImGui::PlotHistogram("##h", data.data(), 256, 0, nullptr,
                             0.0f, 1.0f, ImVec2(histW, histH));
        ImGui::PopID();

        // Image statistics
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.72f, 0.72f, 0.72f, 1.0f));
        ImGui::Text("Media: %.1f   Desv: %.1f", stats.mean, stats.stddev);
        ImGui::Text("Blanco: %.1f%%   Negro: %.1f%%", stats.whitePct, stats.blackPct);
        ImGui::PopStyleColor();

        ImGui::EndGroup();
    };

    drawHist("orig", "Original",
             m_histOrig, m_statsOrig);
    ImGui::SameLine();
    drawHist("A",
             m_labelA.empty() ? "Result" : m_labelA.c_str(),
             m_histA, m_statsA);
    if (hasB) {
        ImGui::SameLine();
        drawHist("B",
                 m_labelB.empty() ? "Result B" : m_labelB.c_str(),
                 m_histB, m_statsB);
    }
}
