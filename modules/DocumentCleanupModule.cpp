#include "DocumentCleanupModule.h"
#include "core/ImageLoader.h"
#include "core/FileDialog.h"
#include "core/HistogramUtils.h"
#include "core/ExportUtils.h"
#include "core/HistogramRenderer.h"
#include "imgui.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cstdio>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────

DocumentCleanupModule::~DocumentCleanupModule()
{
    Utils::deleteTexture(m_texOrig);
    Utils::deleteTexture(m_texResult);
}

// ─── Top-level UI ─────────────────────────────────────────────────────────────

void DocumentCleanupModule::renderUI()
{
    ImGui::PushID("DocumentCleanupModule");

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "Document Cleanup");
    ImGui::Separator();
    ImGui::Spacing();

    // ── File loader ───────────────────────────────────────────────────────────
    if (ImGui::Button("Browse...", ImVec2(80, 0))) {
        std::string p = FileDialog::openImage();
        if (!p.empty()) loadAndProcess(p);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", m_loadedPath.empty() ? "No image loaded" : m_loadedPath.c_str());

    if (!m_loadedPath.empty() && m_perspCorrection) {
        if (m_warpApplied)
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "  Perspective corrected");
        else
            ImGui::TextDisabled("  Page boundary not detected, using full frame");
    }
    ImGui::Spacing();

    // ── Processing toggles ────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextDisabled("PROCESSING OPTIONS");
    ImGui::Spacing();

    if (ImGui::Checkbox("Perspective correction", &m_perspCorrection)) m_needsProcess = true;
    if (ImGui::Checkbox("Shadow reduction (CLAHE)", &m_shadowReduction)) m_needsProcess = true;
    if (ImGui::Checkbox("Sharpening",               &m_sharpenEnabled))  m_needsProcess = true;
    if (ImGui::Checkbox("Binary output",             &m_binaryOut))       m_needsProcess = true;
    if (ImGui::Checkbox("Morphology cleanup",        &m_morphCleanup))    m_needsProcess = true;

    // ── Parameter sliders ─────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextDisabled("PARAMETERS");
    ImGui::Spacing();

    ImGui::Text("Blur strength (odd kernel):");
    if (ImGui::SliderInt("##blur", &m_blurKernel, 1, 15)) {
        if (m_blurKernel % 2 == 0) m_blurKernel++;
        m_needsProcess = true;
    }

    if (m_shadowReduction) {
        ImGui::Text("CLAHE clip limit:");
        if (ImGui::SliderFloat("##clahe", &m_claheClip, 1.0f, 8.0f, "%.1f"))
            m_needsProcess = true;
    }

    if (m_sharpenEnabled) {
        ImGui::Text("Sharpen strength:");
        if (ImGui::SliderFloat("##sharp", &m_sharpenStrength, 0.0f, 2.0f, "%.2f"))
            m_needsProcess = true;
    }

    if (m_binaryOut) {
        ImGui::Text("Adaptive block size (odd):");
        if (ImGui::SliderInt("##blk", &m_adaptBlockSize, 3, 51)) {
            if (m_adaptBlockSize % 2 == 0) m_adaptBlockSize++;
            m_needsProcess = true;
        }
        ImGui::Text("Adaptive C constant:");
        if (ImGui::SliderInt("##adC", &m_adaptC, 0, 30))
            m_needsProcess = true;
    }

    // ── Lazy processing ───────────────────────────────────────────────────────
    if (m_needsProcess && !m_rawBGR.empty()) {
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
    if (m_texResult.valid()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Exportar resultado", ImVec2(140, 0))) {
            std::string path = ExportUtils::nextExportPath("exports", "document_cleanup");
            if (cv::imwrite(path, m_result))
                std::fprintf(stdout, "[DocCleanup] Exportado: %s\n", path.c_str());
        }

        ImGui::SameLine();

        if (ImGui::Button("Exportar histogramas", ImVec2(150, 0))) {
            HistogramRenderer::renderAndSave(
                m_histOrig,
                ExportUtils::nextExportPath("exports", "document_histograma_original"),
                "Original");

            HistogramRenderer::renderAndSave(
                m_histResult,
                ExportUtils::nextExportPath("exports", "document_histograma_limpio"),
                "Limpio");
        }
    }

    ImGui::PopID();
}

// ─── Load & process ───────────────────────────────────────────────────────────

void DocumentCleanupModule::loadAndProcess(const std::string& path)
{
    m_rawBGR = ImageLoader::load(path);
    if (m_rawBGR.empty()) {
        std::fprintf(stderr, "[DocumentCleanup] Failed to load: %s\n", path.c_str());
        return;
    }
    m_loadedPath   = path;
    m_needsProcess = true;
}

// ─── Reprocess ────────────────────────────────────────────────────────────────

void DocumentCleanupModule::reprocess()
{
    if (m_rawBGR.empty()) return;

    // ── Optional perspective correction ───────────────────────────────────────
    cv::Mat workBGR;
    m_warpApplied = false;
    if (m_perspCorrection) {
        cv::Mat warped;
        if (detectPageAndWarp(m_rawBGR, warped)) {
            workBGR       = warped;
            m_warpApplied = true;
        }
    }
    if (workBGR.empty())
        workBGR = m_rawBGR;

    m_original = workBGR;
    Utils::updateTexture(m_texOrig, m_original);

    cv::Mat gray;
    cv::cvtColor(workBGR, gray, cv::COLOR_BGR2GRAY);
    m_histOrig  = HistogramUtils::compute(gray);
    m_statsOrig = HistogramUtils::computeStats(gray);

    // ── Cleanup pipeline ──────────────────────────────────────────────────────
    m_result = applyCleanup(gray);
    Utils::updateTexture(m_texResult, m_result);
    m_histResult  = HistogramUtils::compute(m_result);
    m_statsResult = HistogramUtils::computeStats(m_result);
}

// ─── Perspective detection ────────────────────────────────────────────────────

// Returns {TL, TR, BR, BL} from an unsorted 4-point contour.
static std::vector<cv::Point2f> sortCorners(const std::vector<cv::Point>& pts)
{
    std::vector<cv::Point2f> p(pts.begin(), pts.end());
    std::vector<cv::Point2f> sorted(4);

    std::sort(p.begin(), p.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    });

    sorted[0] = p[0]; // top-left     (minimum x+y)
    sorted[2] = p[3]; // bottom-right (maximum x+y)

    // Of the remaining two: higher x-y → top-right (high x, low y)
    if ((p[1].x - p[1].y) > (p[2].x - p[2].y)) {
        sorted[1] = p[1]; // top-right
        sorted[3] = p[2]; // bottom-left
    } else {
        sorted[1] = p[2]; // top-right
        sorted[3] = p[1]; // bottom-left
    }

    return sorted;
}

bool DocumentCleanupModule::detectPageAndWarp(const cv::Mat& bgr, cv::Mat& out)
{
    // Downscale to max 800px on the longest side for faster edge detection
    const int   maxDim = 800;
    const float scale  = static_cast<float>(maxDim) /
                         static_cast<float>(std::max(bgr.cols, bgr.rows));
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(), scale, scale);

    cv::Mat gray, blurred, edges;
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 50, 150);

    // Dilate to bridge small gaps in document border lines
    cv::Mat dilKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges, dilKernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::sort(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });

    // Require the candidate to cover at least 15% of the frame
    const double minArea = small.cols * small.rows * 0.15;

    for (const auto& c : contours)
    {
        if (cv::contourArea(c) < minArea) break;

        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, cv::arcLength(c, true) * 0.02, true);

        if (approx.size() != 4 || !cv::isContourConvex(approx)) continue;

        // Scale corners back to full-resolution coordinates
        std::vector<cv::Point> fullApprox;
        fullApprox.reserve(4);
        for (const auto& pt : approx)
            fullApprox.push_back(cv::Point(
                static_cast<int>(std::round(pt.x / scale)),
                static_cast<int>(std::round(pt.y / scale))));

        auto src = sortCorners(fullApprox);

        float w1 = static_cast<float>(cv::norm(src[1] - src[0]));
        float w2 = static_cast<float>(cv::norm(src[2] - src[3]));
        float h1 = static_cast<float>(cv::norm(src[3] - src[0]));
        float h2 = static_cast<float>(cv::norm(src[2] - src[1]));
        int W = static_cast<int>(std::max(w1, w2));
        int H = static_cast<int>(std::max(h1, h2));

        if (W < 100 || H < 100) continue;

        std::vector<cv::Point2f> dst = {
            {0.0f,       0.0f      },
            {(float)W-1, 0.0f      },
            {(float)W-1, (float)H-1},
            {0.0f,       (float)H-1}
        };

        cv::Mat M = cv::getPerspectiveTransform(src, dst);
        cv::warpPerspective(bgr, out, M, cv::Size(W, H));
        return true;
    }

    return false;
}

// ─── Cleanup pipeline ─────────────────────────────────────────────────────────

cv::Mat DocumentCleanupModule::applyCleanup(const cv::Mat& gray)
{
    cv::Mat img = gray.clone();

    // 1. Illumination normalization — CLAHE before blur preserves local detail
    if (m_shadowReduction) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
            static_cast<double>(m_claheClip), cv::Size(8, 8));
        clahe->apply(img, img);
    }

    // 2. Noise reduction
    if (m_blurKernel > 1) {
        int bk = m_blurKernel;
        if (bk % 2 == 0) bk++;
        cv::GaussianBlur(img, img, cv::Size(bk, bk), 0);
    }

    // 3. Unsharp mask — sharpens text edges before threshold or grayscale output
    if (m_sharpenEnabled && m_sharpenStrength > 0.0f) {
        cv::Mat blurred;
        cv::GaussianBlur(img, blurred, cv::Size(0, 0), 3.0);
        cv::addWeighted(img,    1.0 + static_cast<double>(m_sharpenStrength),
                        blurred, -static_cast<double>(m_sharpenStrength),
                        0.0, img);
    }

    // 4. Final output
    if (m_binaryOut) {
        int blockSize = m_adaptBlockSize;
        if (blockSize < 3)      blockSize = 3;
        if (blockSize % 2 == 0) blockSize++;

        cv::adaptiveThreshold(img, img, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY,
                              blockSize,
                              static_cast<double>(m_adaptC));

        // Remove isolated noise dots from the binarized result
        if (m_morphCleanup) {
            cv::Mat mk = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::morphologyEx(img, img, cv::MORPH_OPEN, mk);
        }
    }

    return img;
}

// ─── Image row renderer ───────────────────────────────────────────────────────

void DocumentCleanupModule::drawImageRow()
{
    if (!m_texOrig.valid()) {
        ImGui::TextDisabled("No image loaded. Click Browse to open an image.");
        return;
    }

    float avail   = ImGui::GetContentRegionAvail().x;
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float imgW    = (avail - spacing) / 2.0f;
    float aspect  = static_cast<float>(m_texOrig.height) /
                    static_cast<float>(m_texOrig.width);
    float imgH    = imgW * aspect;

    ImGui::Text("Original");
    ImGui::SameLine(imgW + spacing);
    ImGui::Text("Cleaned Result");

    ImGui::Image((ImTextureID)(uintptr_t)m_texOrig.id,   ImVec2(imgW, imgH));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)(uintptr_t)m_texResult.id, ImVec2(imgW, imgH));
}

// ─── Histogram renderer ───────────────────────────────────────────────────────

void DocumentCleanupModule::drawHistograms()
{
    if (m_rawBGR.empty()) return;

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

    drawHist("orig",   "Original",      m_histOrig,   m_statsOrig);
    ImGui::SameLine();
    drawHist("result", "Cleaned Result", m_histResult, m_statsResult);
}
