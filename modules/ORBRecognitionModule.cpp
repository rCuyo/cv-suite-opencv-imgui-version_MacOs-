// core/gl.h must be included before GLFW when GLFW_INCLUDE_NONE is active
#include "core/gl.h"
#include <GLFW/glfw3.h>

#include "imgui.h"

#include "ORBRecognitionModule.h"
#include "core/FileDialog.h"
#include "core/ExportUtils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>

// ─── Destructor ───────────────────────────────────────────────────────────────

ORBRecognitionModule::~ORBRecognitionModule()
{
    Utils::deleteTexture(m_texRef);
    Utils::deleteTexture(m_texQuery);
    Utils::deleteTexture(m_texMatches);
}

// ─── Image loading ────────────────────────────────────────────────────────────

void ORBRecognitionModule::loadReference(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) return;
    m_rawRef        = img;
    m_refPath       = path;
    m_hasRef        = true;
    m_hasFeatures   = false;
    m_hasMatches    = false;
    m_imgRefDisplay = img.clone();
    Utils::updateTexture(m_texRef, m_imgRefDisplay);
    m_needsProcess  = true;
}

void ORBRecognitionModule::loadQuery(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) return;
    m_rawQuery        = img;
    m_queryPath       = path;
    m_hasQuery        = true;
    m_hasFeatures     = false;
    m_hasMatches      = false;
    m_imgQueryDisplay = img.clone();
    Utils::updateTexture(m_texQuery, m_imgQueryDisplay);
    m_needsProcess    = true;
}

// ─── Processing ───────────────────────────────────────────────────────────────

void ORBRecognitionModule::reprocess()
{
    detectFeatures();
    if (m_hasFeatures) matchFeatures();
    buildVisuals();
    m_needsProcess = false;
}

void ORBRecognitionModule::detectFeatures()
{
    m_kpRef.clear();
    m_kpQuery.clear();
    m_descRef.release();
    m_descQuery.release();
    m_hasFeatures  = false;
    m_kpRefCount   = 0;
    m_kpQueryCount = 0;

    // edgeThreshold and patchSize are kept at ORB defaults (31)
    auto orb = cv::ORB::create(m_nFeatures, m_scaleFactor, m_nLevels);

    auto t0 = std::chrono::high_resolution_clock::now();

    if (m_hasRef && !m_rawRef.empty()) {
        cv::Mat gray;
        cv::cvtColor(m_rawRef, gray, cv::COLOR_BGR2GRAY);
        orb->detectAndCompute(gray, cv::noArray(), m_kpRef, m_descRef);
        m_kpRefCount = static_cast<int>(m_kpRef.size());
    }

    if (m_hasQuery && !m_rawQuery.empty()) {
        cv::Mat gray;
        cv::cvtColor(m_rawQuery, gray, cv::COLOR_BGR2GRAY);
        orb->detectAndCompute(gray, cv::noArray(), m_kpQuery, m_descQuery);
        m_kpQueryCount = static_cast<int>(m_kpQuery.size());
    }

    auto t1        = std::chrono::high_resolution_clock::now();
    m_detectTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    m_hasFeatures = (m_hasRef && m_hasQuery
                     && !m_descRef.empty() && !m_descQuery.empty());
}

void ORBRecognitionModule::matchFeatures()
{
    m_goodMatches.clear();
    m_hasMatches = false;
    m_matchCount = 0;
    if (!m_hasFeatures) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    // ORB uses binary descriptors → Hamming distance; crossCheck ensures
    // mutual consistency (each ref KP's best match also maps back to it)
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> allMatches;
    matcher->match(m_descRef, m_descQuery, allMatches);

    // Sort by distance and keep only those within the Hamming threshold
    std::sort(allMatches.begin(), allMatches.end());
    for (auto& m : allMatches) {
        if (m.distance <= static_cast<float>(m_matchThreshold))
            m_goodMatches.push_back(m);
    }

    auto t1      = std::chrono::high_resolution_clock::now();
    m_matchTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    m_matchCount  = static_cast<int>(m_goodMatches.size());
    m_hasMatches  = true;
}

void ORBRecognitionModule::buildVisuals()
{
    const int kMaxKPDim    = 900;
    const int kMaxMatchDim = 700;

    auto makeKPImage = [](const cv::Mat& src,
                          const std::vector<cv::KeyPoint>& kps,
                          const cv::Scalar& color,
                          int maxDim,
                          cv::Mat& out)
    {
        if (src.empty()) return;
        float s = std::min(1.0f,
                           static_cast<float>(maxDim) / static_cast<float>(std::max(src.cols, src.rows)));
        cv::Mat small;
        if (s < 0.999f) cv::resize(src, small, cv::Size(), s, s, cv::INTER_AREA);
        else            small = src.clone();

        if (kps.empty()) { out = small; return; }

        std::vector<cv::KeyPoint> kpsS;
        kpsS.reserve(kps.size());
        for (auto kp : kps) {
            kp.pt.x *= s; kp.pt.y *= s; kp.size *= s;
            kpsS.push_back(kp);
        }
        cv::drawKeypoints(small, kpsS, out, color,
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    };

    // ORB keypoints: orange-yellow for reference, cyan for comparison
    if (m_hasRef)
        makeKPImage(m_rawRef,   m_kpRef,   cv::Scalar(0, 160, 255), kMaxKPDim, m_imgRefDisplay);
    if (m_hasQuery)
        makeKPImage(m_rawQuery, m_kpQuery, cv::Scalar(255, 220, 0), kMaxKPDim, m_imgQueryDisplay);

    if (m_hasRef)   Utils::updateTexture(m_texRef,   m_imgRefDisplay);
    if (m_hasQuery) Utils::updateTexture(m_texQuery, m_imgQueryDisplay);

    if (m_hasMatches && m_hasRef && m_hasQuery) {
        auto scaleImg = [](const cv::Mat& src, int maxDim, cv::Mat& out) -> float {
            float s = std::min(1.0f,
                               static_cast<float>(maxDim) / static_cast<float>(std::max(src.cols, src.rows)));
            if (s < 0.999f) cv::resize(src, out, cv::Size(), s, s, cv::INTER_AREA);
            else            out = src.clone();
            return s;
        };

        auto scaleKPs = [](const std::vector<cv::KeyPoint>& kps, float s) {
            std::vector<cv::KeyPoint> out;
            out.reserve(kps.size());
            for (auto kp : kps) {
                kp.pt.x *= s; kp.pt.y *= s; kp.size *= s;
                out.push_back(kp);
            }
            return out;
        };

        cv::Mat refS, queryS;
        float sR = scaleImg(m_rawRef,   kMaxMatchDim, refS);
        float sQ = scaleImg(m_rawQuery, kMaxMatchDim, queryS);

        auto kpRefS   = scaleKPs(m_kpRef,   sR);
        auto kpQueryS = scaleKPs(m_kpQuery, sQ);

        // Show best 100 matches for visual clarity
        auto displayMatches = m_goodMatches;
        if (static_cast<int>(displayMatches.size()) > 100)
            displayMatches.resize(100);

        cv::drawMatches(refS, kpRefS, queryS, kpQueryS,
                        displayMatches, m_imgMatches,
                        cv::Scalar(50, 230, 255), cv::Scalar(255, 100, 0),
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        Utils::updateTexture(m_texMatches, m_imgMatches);
    }
}

// ─── Export ───────────────────────────────────────────────────────────────────

void ORBRecognitionModule::exportImage(const cv::Mat& img, const std::string& prefix)
{
    if (img.empty()) return;
    std::string path = ExportUtils::nextExportPath("exports", prefix, "png");
    if (cv::imwrite(path, img)) {
        m_exportMsg     = "Exportado: " + path;
        m_exportMsgTime = glfwGetTime();
    }
}

// ─── Reset ────────────────────────────────────────────────────────────────────

void ORBRecognitionModule::reset()
{
    Utils::deleteTexture(m_texRef);
    Utils::deleteTexture(m_texQuery);
    Utils::deleteTexture(m_texMatches);
    m_rawRef = m_rawQuery = cv::Mat();
    m_imgRefDisplay = m_imgQueryDisplay = m_imgMatches = cv::Mat();
    m_kpRef.clear(); m_kpQuery.clear();
    m_descRef.release(); m_descQuery.release();
    m_goodMatches.clear();
    m_hasRef = m_hasQuery = m_hasFeatures = m_hasMatches = m_needsProcess = false;
    m_kpRefCount = m_kpQueryCount = m_matchCount = 0;
    m_detectTimeMs = m_matchTimeMs = 0.0f;
    m_refPath.clear(); m_queryPath.clear();
    m_exportMsg.clear();
}

// ─── UI ──────────────────────────────────────────────────────────────────────

void ORBRecognitionModule::renderUI()
{
    ImGui::PushID("ORBRecognition");

    if (m_needsProcess) reprocess();

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.75f, 1.0f), "ORB Object Recognition");
    ImGui::TextDisabled(
        "Reconocimiento rapido de objetos con descriptores binarios — ideal para hardware limitado.");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Load / Reset buttons ──────────────────────────────────────────────────
    if (ImGui::Button("Ref. Image##orb", ImVec2(140, 0))) {
        auto p = FileDialog::openImage();
        if (!p.empty()) loadReference(p);
    }
    ImGui::SameLine();
    if (ImGui::Button("Comparison Image##orb", ImVec2(155, 0))) {
        auto p = FileDialog::openImage();
        if (!p.empty()) loadQuery(p);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset##orb", ImVec2(60, 0)))
        reset();

    if (!m_refPath.empty())   ImGui::TextDisabled("  Ref: %s", m_refPath.c_str());
    if (!m_queryPath.empty()) ImGui::TextDisabled("  Cmp: %s", m_queryPath.c_str());

    ImGui::Spacing();

    // ── Action buttons ────────────────────────────────────────────────────────
    bool canDetect    = m_hasRef || m_hasQuery;
    bool canMatch     = m_hasFeatures;
    bool canExportMat = m_hasMatches && !m_imgMatches.empty();
    bool canExportKPR = !m_imgRefDisplay.empty()   && m_kpRefCount   > 0;
    bool canExportKPQ = !m_imgQueryDisplay.empty() && m_kpQueryCount > 0;

    if (!canDetect) ImGui::BeginDisabled();
    if (ImGui::Button("Detect Features##orb", ImVec2(140, 0))) {
        detectFeatures();
        if (m_hasFeatures) matchFeatures();
        buildVisuals();
    }
    if (!canDetect) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!canMatch) ImGui::BeginDisabled();
    if (ImGui::Button("Match Features##orb", ImVec2(125, 0))) {
        matchFeatures();
        buildVisuals();
    }
    if (!canMatch) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!canExportMat) ImGui::BeginDisabled();
    if (ImGui::Button("Export Matches##orb", ImVec2(120, 0)))
        exportImage(m_imgMatches, "orb_matches");
    if (!canExportMat) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!canExportKPR) ImGui::BeginDisabled();
    if (ImGui::Button("Export KP Ref##orb", ImVec2(108, 0)))
        exportImage(m_imgRefDisplay, "orb_kp_ref");
    if (!canExportKPR) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!canExportKPQ) ImGui::BeginDisabled();
    if (ImGui::Button("Export KP Cmp##orb", ImVec2(110, 0)))
        exportImage(m_imgQueryDisplay, "orb_kp_query");
    if (!canExportKPQ) ImGui::EndDisabled();

    if (!m_exportMsg.empty()) {
        if (glfwGetTime() - m_exportMsgTime < 3.0)
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "%s", m_exportMsg.c_str());
        else
            m_exportMsg.clear();
    }

    ImGui::Spacing();
    ImGui::Separator();

    // ── Parameters ────────────────────────────────────────────────────────────
    ImGui::Text("Parametros ORB:");
    ImGui::Spacing();

    bool changed = false;

    ImGui::SetNextItemWidth(180);
    changed |= ImGui::SliderInt("Num. Features##orb", &m_nFeatures, 50, 2000);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Cantidad maxima de keypoints ORB detectados.");

    ImGui::SameLine(0, 24);
    ImGui::SetNextItemWidth(180);
    changed |= ImGui::SliderFloat("Scale Factor##orb", &m_scaleFactor, 1.1f, 2.0f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Factor de escala entre niveles de la piramide de imagen.");

    ImGui::SetNextItemWidth(180);
    changed |= ImGui::SliderInt("Pyramid Levels##orb", &m_nLevels, 1, 12);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Niveles de la piramide: mas niveles = invariancia a mayor escala.");

    ImGui::SameLine(0, 24);
    ImGui::SetNextItemWidth(180);
    changed |= ImGui::SliderInt("Match Threshold##orb", &m_matchThreshold, 0, 256);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Distancia Hamming maxima permitida (bits diferentes en el descriptor).\n"
                          "Valor menor = matches mas estrictos. ORB tiene 256 bits por descriptor.");

    if (changed && canDetect)
        m_needsProcess = true;

    ImGui::Spacing();
    ImGui::Separator();

    // ── Statistics & recognition result ───────────────────────────────────────
    if (m_hasRef || m_hasQuery) {
        ImGui::Text("Estadisticas:");
        ImGui::Spacing();
        if (m_hasRef)
            ImGui::Text("Referencia :  %d x %d px   |   Keypoints ORB: %d",
                        m_rawRef.cols, m_rawRef.rows, m_kpRefCount);
        if (m_hasQuery)
            ImGui::Text("Comparacion:  %d x %d px   |   Keypoints ORB: %d",
                        m_rawQuery.cols, m_rawQuery.rows, m_kpQueryCount);

        if (m_hasMatches) {
            int   refKP = std::max(m_kpRefCount, 1);
            float pct   = 100.0f * static_cast<float>(m_matchCount) / static_cast<float>(refKP);
            ImGui::Text("Matches validos (Hamming <= %d): %d   (%.1f%% de KP ref.)",
                        m_matchThreshold, m_matchCount, pct);
            ImGui::Spacing();
            if (m_matchCount >= 10)
                ImGui::TextColored(ImVec4(0.20f, 1.00f, 0.30f, 1.0f),
                    "OBJETO RECONOCIDO  —  %d matches validos", m_matchCount);
            else
                ImGui::TextColored(ImVec4(1.00f, 0.40f, 0.30f, 1.0f),
                    "No reconocido  —  %d matches (minimo recomendado: 10)", m_matchCount);
        }

        if (m_detectTimeMs > 0.0f)
            ImGui::TextDisabled("Tiempo  —  deteccion: %.1f ms   matching: %.1f ms",
                                m_detectTimeMs, m_matchTimeMs);

        ImGui::Spacing();
        ImGui::Separator();
    }

    // ── Academic explanation ──────────────────────────────────────────────────
    ImGui::Spacing();
    if (ImGui::TreeNode("Acerca de ORB##tree")) {
        ImGui::Spacing();
        ImGui::TextWrapped(
            "ORB combina FAST y BRIEF para crear un descriptor rapido y eficiente. "
            "ORB es una alternativa ideal para dispositivos con hardware limitado "
            "debido a su bajo costo computacional.");
        ImGui::Spacing();
        ImGui::BulletText("Mas rapido que SIFT: usa FAST en lugar del costoso calculo gaussiano.");
        ImGui::BulletText("Bajo costo computacional: descriptores binarios (256 bits).");
        ImGui::BulletText("Matching eficiente: distancia Hamming calculable con XOR + popcount.");
        ImGui::BulletText("Menor precision en algunos casos: los descriptores binarios son");
        ImGui::BulletText("  menos discriminativos que los flotantes de SIFT (128 floats).");
        ImGui::BulletText("Invariante a rotacion: ORB estima la orientacion del patch con");
        ImGui::BulletText("  el centroide de intensidad (Intensity Centroid).");
        ImGui::Spacing();
        ImGui::TextWrapped(
            "Matching: BFMatcher con NORM_HAMMING + crossCheck. CrossCheck garantiza "
            "que el match sea mutuamente consistente (A->B y B->A coinciden).");
        ImGui::Spacing();
        ImGui::TextDisabled("Referencia: Rublee et al., \"ORB: An efficient alternative");
        ImGui::TextDisabled("to SIFT or SURF\", ICCV 2011.");
        ImGui::TreePop();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ── Image display ─────────────────────────────────────────────────────────
    float contentW = ImGui::GetContentRegionAvail().x;

    if (!m_hasRef && !m_hasQuery) {
        ImGui::TextDisabled(
            "Carga una imagen de referencia y una de comparacion para comenzar.");
        ImGui::PopID();
        return;
    }

    if (m_texRef.valid() || m_texQuery.valid()) {
        float panelW = (contentW - 16.0f) * 0.5f;
        float maxH   = 320.0f;

        ImGui::BeginGroup();
        ImGui::TextDisabled("Referencia%s", m_kpRefCount > 0 ? "  (keypoints naranja)" : "");
        if (m_texRef.valid()) {
            float s = std::min(panelW / static_cast<float>(m_texRef.width),
                               maxH   / static_cast<float>(m_texRef.height));
            ImGui::Image((ImTextureID)(uintptr_t)m_texRef.id,
                         ImVec2(m_texRef.width * s, m_texRef.height * s));
        } else {
            ImGui::Dummy(ImVec2(panelW, 200.0f));
            ImGui::TextDisabled("(sin imagen)");
        }
        ImGui::EndGroup();

        ImGui::SameLine(0, 16);

        ImGui::BeginGroup();
        ImGui::TextDisabled("Comparacion%s", m_kpQueryCount > 0 ? "  (keypoints amarillo)" : "");
        if (m_texQuery.valid()) {
            float s = std::min(panelW / static_cast<float>(m_texQuery.width),
                               maxH   / static_cast<float>(m_texQuery.height));
            ImGui::Image((ImTextureID)(uintptr_t)m_texQuery.id,
                         ImVec2(m_texQuery.width * s, m_texQuery.height * s));
        } else {
            ImGui::Dummy(ImVec2(panelW, 200.0f));
            ImGui::TextDisabled("(sin imagen)");
        }
        ImGui::EndGroup();
    }

    if (m_texMatches.valid()) {
        ImGui::Spacing();
        ImGui::TextDisabled("Correspondencias  (Hamming <= %d, max. 100 mostrados):",
                            m_matchThreshold);
        float maxH  = 420.0f;
        float s     = std::min(contentW / static_cast<float>(m_texMatches.width),
                               maxH     / static_cast<float>(m_texMatches.height));
        ImGui::Image((ImTextureID)(uintptr_t)m_texMatches.id,
                     ImVec2(m_texMatches.width * s, m_texMatches.height * s));
    }

    ImGui::PopID();
}
