// core/gl.h must be included before GLFW when GLFW_INCLUDE_NONE is active
#include "core/gl.h"
#include <GLFW/glfw3.h>

#include "imgui.h"

#include "SIFTRecognitionModule.h"
#include "core/FileDialog.h"
#include "core/ExportUtils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>

// ─── Destructor ───────────────────────────────────────────────────────────────

SIFTRecognitionModule::~SIFTRecognitionModule()
{
    Utils::deleteTexture(m_texRef);
    Utils::deleteTexture(m_texQuery);
    Utils::deleteTexture(m_texMatches);
}

// ─── Image loading ────────────────────────────────────────────────────────────

void SIFTRecognitionModule::loadReference(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) return;
    m_rawRef        = img;
    m_refPath       = path;
    m_hasRef        = true;
    m_hasFeatures   = false;
    m_hasMatches    = false;
    // Show raw image immediately while processing
    m_imgRefDisplay = img.clone();
    Utils::updateTexture(m_texRef, m_imgRefDisplay);
    m_needsProcess  = true;
}

void SIFTRecognitionModule::loadQuery(const std::string& path)
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

void SIFTRecognitionModule::reprocess()
{
    detectFeatures();
    if (m_hasFeatures) matchFeatures();
    buildVisuals();
    m_needsProcess = false;
}

void SIFTRecognitionModule::detectFeatures()
{
    m_kpRef.clear();
    m_kpQuery.clear();
    m_descRef.release();
    m_descQuery.release();
    m_hasFeatures    = false;
    m_kpRefCount     = 0;
    m_kpQueryCount   = 0;

    auto sift = cv::SIFT::create(m_nFeatures, 3,
                                  static_cast<double>(m_contrastThreshold),
                                  static_cast<double>(m_edgeThreshold),
                                  1.6);

    auto t0 = std::chrono::high_resolution_clock::now();

    if (m_hasRef && !m_rawRef.empty()) {
        cv::Mat gray;
        cv::cvtColor(m_rawRef, gray, cv::COLOR_BGR2GRAY);
        sift->detectAndCompute(gray, cv::noArray(), m_kpRef, m_descRef);
        m_kpRefCount = static_cast<int>(m_kpRef.size());
    }

    if (m_hasQuery && !m_rawQuery.empty()) {
        cv::Mat gray;
        cv::cvtColor(m_rawQuery, gray, cv::COLOR_BGR2GRAY);
        sift->detectAndCompute(gray, cv::noArray(), m_kpQuery, m_descQuery);
        m_kpQueryCount = static_cast<int>(m_kpQuery.size());
    }

    auto t1        = std::chrono::high_resolution_clock::now();
    m_detectTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // Both images need descriptors before matching is possible
    m_hasFeatures = (m_hasRef && m_hasQuery
                     && !m_descRef.empty() && !m_descQuery.empty());
}

void SIFTRecognitionModule::matchFeatures()
{
    m_goodMatches.clear();
    m_hasMatches = false;
    m_matchCount = 0;
    if (!m_hasFeatures) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    auto matcher = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(m_descRef, m_descQuery, knnMatches, 2);

    // Lowe ratio test: accept only unambiguous matches
    for (auto& m : knnMatches) {
        if (m.size() >= 2 && m[0].distance < m_ratioThresh * m[1].distance)
            m_goodMatches.push_back(m[0]);
    }

    auto t1      = std::chrono::high_resolution_clock::now();
    m_matchTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    m_matchCount  = static_cast<int>(m_goodMatches.size());
    m_hasMatches  = true;
}

void SIFTRecognitionModule::buildVisuals()
{
    // Max image dimension for keypoint visualization textures
    const int kMaxKPDim    = 900;
    // Max image dimension per side for the matches strip
    const int kMaxMatchDim = 700;

    // Scale image + keypoints together, then draw rich keypoints (circle + orientation)
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

        std::vector<cv::KeyPoint> kpsScaled;
        kpsScaled.reserve(kps.size());
        for (auto kp : kps) {
            kp.pt.x *= s;
            kp.pt.y *= s;
            kp.size  *= s;
            kpsScaled.push_back(kp);
        }
        cv::drawKeypoints(small, kpsScaled, out, color,
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    };

    if (m_hasRef)
        makeKPImage(m_rawRef,   m_kpRef,   cv::Scalar(0, 220, 0),   kMaxKPDim, m_imgRefDisplay);
    if (m_hasQuery)
        makeKPImage(m_rawQuery, m_kpQuery, cv::Scalar(0, 180, 255), kMaxKPDim, m_imgQueryDisplay);

    if (m_hasRef)   Utils::updateTexture(m_texRef,   m_imgRefDisplay);
    if (m_hasQuery) Utils::updateTexture(m_texQuery, m_imgQueryDisplay);

    // Matches visualization: scale images, scale keypoints, draw match lines
    if (m_hasMatches && m_hasRef && m_hasQuery) {
        auto scaleImage = [](const cv::Mat& src, int maxDim, cv::Mat& out) -> float {
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
                kp.pt.x *= s;
                kp.pt.y *= s;
                kp.size  *= s;
                out.push_back(kp);
            }
            return out;
        };

        cv::Mat refSmall, querySmall;
        float sR = scaleImage(m_rawRef,   kMaxMatchDim, refSmall);
        float sQ = scaleImage(m_rawQuery, kMaxMatchDim, querySmall);

        auto kpRefS   = scaleKPs(m_kpRef,   sR);
        auto kpQueryS = scaleKPs(m_kpQuery, sQ);

        // Show the best 100 matches for visual clarity
        auto displayMatches = m_goodMatches;
        std::sort(displayMatches.begin(), displayMatches.end(),
                  [](const cv::DMatch& a, const cv::DMatch& b) {
                      return a.distance < b.distance;
                  });
        if (static_cast<int>(displayMatches.size()) > 100)
            displayMatches.resize(100);

        cv::drawMatches(refSmall,   kpRefS,
                        querySmall, kpQueryS,
                        displayMatches, m_imgMatches,
                        cv::Scalar(50, 240, 50),   // match line color
                        cv::Scalar(255, 100, 0),   // single-point color (not drawn)
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        Utils::updateTexture(m_texMatches, m_imgMatches);
    }
}

// ─── Export ───────────────────────────────────────────────────────────────────

void SIFTRecognitionModule::exportImage(const cv::Mat& img, const std::string& prefix)
{
    if (img.empty()) return;
    std::string path = ExportUtils::nextExportPath("exports", prefix, "png");
    if (cv::imwrite(path, img)) {
        m_exportMsg     = "Exportado: " + path;
        m_exportMsgTime = glfwGetTime();
    }
}

// ─── Reset ────────────────────────────────────────────────────────────────────

void SIFTRecognitionModule::reset()
{
    Utils::deleteTexture(m_texRef);
    Utils::deleteTexture(m_texQuery);
    Utils::deleteTexture(m_texMatches);

    m_rawRef = m_rawQuery = cv::Mat();
    m_imgRefDisplay = m_imgQueryDisplay = m_imgMatches = cv::Mat();
    m_kpRef.clear();
    m_kpQuery.clear();
    m_descRef.release();
    m_descQuery.release();
    m_goodMatches.clear();

    m_hasRef = m_hasQuery = m_hasFeatures = m_hasMatches = m_needsProcess = false;
    m_kpRefCount = m_kpQueryCount = m_matchCount = 0;
    m_detectTimeMs = m_matchTimeMs = 0.0f;
    m_refPath.clear();
    m_queryPath.clear();
    m_exportMsg.clear();
}

// ─── UI ──────────────────────────────────────────────────────────────────────

void SIFTRecognitionModule::renderUI()
{
    ImGui::PushID("SIFTRecognition");

    // Consume the deferred reprocess flag (set on load or param change)
    if (m_needsProcess) reprocess();

    // ── Header ────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.75f, 1.0f), "SIFT Object Recognition");
    ImGui::TextDisabled(
        "Reconocimiento de objetos invariante a escala, rotacion, orientacion e iluminacion.");
    ImGui::Separator();
    ImGui::Spacing();

    // ── Load / Reset buttons ──────────────────────────────────────────────────
    if (ImGui::Button("Ref. Image##sift", ImVec2(140, 0))) {
        auto p = FileDialog::openImage();
        if (!p.empty()) loadReference(p);
    }
    ImGui::SameLine();
    if (ImGui::Button("Comparison Image##sift", ImVec2(155, 0))) {
        auto p = FileDialog::openImage();
        if (!p.empty()) loadQuery(p);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset##sift", ImVec2(60, 0)))
        reset();

    // File path hints
    if (!m_refPath.empty())
        ImGui::TextDisabled("  Ref: %s", m_refPath.c_str());
    if (!m_queryPath.empty())
        ImGui::TextDisabled("  Cmp: %s", m_queryPath.c_str());

    ImGui::Spacing();

    // ── Action buttons ────────────────────────────────────────────────────────
    bool canDetect    = m_hasRef || m_hasQuery;
    bool canMatch     = m_hasFeatures;
    bool canExportMat = m_hasMatches && !m_imgMatches.empty();
    bool canExportKPR = !m_imgRefDisplay.empty()   && m_kpRefCount   > 0;
    bool canExportKPQ = !m_imgQueryDisplay.empty() && m_kpQueryCount > 0;

    if (!canDetect) ImGui::BeginDisabled();
    if (ImGui::Button("Detect Features##sift", ImVec2(140, 0))) {
        detectFeatures();
        if (m_hasFeatures) matchFeatures();
        buildVisuals();
    }
    if (!canDetect) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!canMatch) ImGui::BeginDisabled();
    if (ImGui::Button("Match Features##sift", ImVec2(125, 0))) {
        matchFeatures();
        buildVisuals();
    }
    if (!canMatch) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!canExportMat) ImGui::BeginDisabled();
    if (ImGui::Button("Export Matches##sift", ImVec2(120, 0)))
        exportImage(m_imgMatches, "sift_matches");
    if (!canExportMat) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!canExportKPR) ImGui::BeginDisabled();
    if (ImGui::Button("Export KP Ref##sift", ImVec2(108, 0)))
        exportImage(m_imgRefDisplay, "sift_kp_ref");
    if (!canExportKPR) ImGui::EndDisabled();

    ImGui::SameLine();
    if (!canExportKPQ) ImGui::BeginDisabled();
    if (ImGui::Button("Export KP Cmp##sift", ImVec2(110, 0)))
        exportImage(m_imgQueryDisplay, "sift_kp_query");
    if (!canExportKPQ) ImGui::EndDisabled();

    // Timed export confirmation
    if (!m_exportMsg.empty()) {
        if (glfwGetTime() - m_exportMsgTime < 3.0)
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "%s", m_exportMsg.c_str());
        else
            m_exportMsg.clear();
    }

    ImGui::Spacing();
    ImGui::Separator();

    // ── Parameters ────────────────────────────────────────────────────────────
    ImGui::Text("Parametros SIFT:");
    ImGui::Spacing();

    bool changed = false;

    ImGui::SetNextItemWidth(180);
    changed |= ImGui::SliderInt("Num. Keypoints##sift", &m_nFeatures, 50, 5000);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Cantidad maxima de keypoints detectados por imagen.");

    ImGui::SameLine(0, 24);
    ImGui::SetNextItemWidth(180);
    changed |= ImGui::SliderFloat("Contrast Threshold##sift",
                                   &m_contrastThreshold, 0.01f, 0.20f, "%.3f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Umbral de contraste: valores bajos detectan mas keypoints en zonas debiles.");

    ImGui::SetNextItemWidth(180);
    changed |= ImGui::SliderFloat("Edge Threshold##sift",
                                   &m_edgeThreshold, 1.0f, 50.0f, "%.1f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Filtrado de bordes debiles: valores altos eliminan mas respuestas de borde.");

    ImGui::SameLine(0, 24);
    ImGui::SetNextItemWidth(180);
    changed |= ImGui::SliderFloat("Match Dist. (Lowe)##sift",
                                   &m_ratioThresh, 0.50f, 0.95f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Lowe ratio test: 0.75 es el valor clasico de D. Lowe (2004).\n"
                          "Valores bajos = matches mas estrictos y confiables.");

    if (changed && canDetect)
        m_needsProcess = true;

    ImGui::Spacing();
    ImGui::Separator();

    // ── Statistics & recognition result ───────────────────────────────────────
    if (m_hasRef || m_hasQuery) {
        ImGui::Text("Estadisticas:");
        ImGui::Spacing();

        if (m_hasRef)
            ImGui::Text("Referencia :  %d x %d px   |   Keypoints SIFT: %d",
                        m_rawRef.cols, m_rawRef.rows, m_kpRefCount);
        if (m_hasQuery)
            ImGui::Text("Comparacion:  %d x %d px   |   Keypoints SIFT: %d",
                        m_rawQuery.cols, m_rawQuery.rows, m_kpQueryCount);

        if (m_hasMatches) {
            int   refKP = std::max(m_kpRefCount, 1);
            float pct   = 100.0f * static_cast<float>(m_matchCount) / static_cast<float>(refKP);
            ImGui::Text("Matches validos (Lowe): %d   (%.1f%% de KP referencia)",
                        m_matchCount, pct);

            ImGui::Spacing();
            // Recognition verdict — 10 good matches is the standard literature threshold
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

    // ── Academic explanation (collapsible) ────────────────────────────────────
    ImGui::Spacing();
    if (ImGui::TreeNode("Acerca de SIFT##tree")) {
        ImGui::Spacing();
        ImGui::TextWrapped(
            "SIFT detecta puntos clave invariantes a escala y rotacion. "
            "Usa gradientes locales de imagen para generar descriptores robustos "
            "de 128 dimensiones que son comparables entre imagenes con transformaciones.");
        ImGui::Spacing();
        ImGui::BulletText("Invariante a rotacion: cada keypoint tiene orientacion canonica.");
        ImGui::BulletText("Invariante a escala: detecta features en piramide Gaussiana multi-escala.");
        ImGui::BulletText("Robusto a iluminacion: el descriptor se normaliza en norma L2.");
        ImGui::BulletText("Parcialmente robusto a perspectiva: tolera cambios moderados de viewpoint.");
        ImGui::Spacing();
        ImGui::TextWrapped(
            "SIFT puede manejar variabilidad de imagenes debido a que sus descriptores "
            "son invariantes a escala y orientacion. El Lowe Ratio Test (D. Lowe, 2004) "
            "garantiza que solo se acepten matches no ambiguos: un match es valido si "
            "su distancia es significativamente menor al segundo mejor candidato.");
        ImGui::Spacing();
        ImGui::TextDisabled("Referencia: D. Lowe, \"Distinctive Image Features from");
        ImGui::TextDisabled("Scale-Invariant Keypoints\", IJCV 2004.");
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

    // Two columns: reference | comparison
    if (m_texRef.valid() || m_texQuery.valid()) {
        float panelW = (contentW - 16.0f) * 0.5f;
        float maxH   = 320.0f;

        // Reference panel
        ImGui::BeginGroup();
        ImGui::TextDisabled("Imagen Referencia%s",
                            m_kpRefCount > 0 ? "  (keypoints en verde)" : "");
        if (m_texRef.valid()) {
            float s = std::min(panelW  / static_cast<float>(m_texRef.width),
                               maxH    / static_cast<float>(m_texRef.height));
            ImGui::Image((ImTextureID)(uintptr_t)m_texRef.id,
                         ImVec2(m_texRef.width * s, m_texRef.height * s));
        } else {
            ImGui::Dummy(ImVec2(panelW, 200.0f));
            ImGui::TextDisabled("(sin imagen)");
        }
        ImGui::EndGroup();

        ImGui::SameLine(0, 16);

        // Comparison panel
        ImGui::BeginGroup();
        ImGui::TextDisabled("Imagen Comparacion%s",
                            m_kpQueryCount > 0 ? "  (keypoints en naranja)" : "");
        if (m_texQuery.valid()) {
            float s = std::min(panelW  / static_cast<float>(m_texQuery.width),
                               maxH    / static_cast<float>(m_texQuery.height));
            ImGui::Image((ImTextureID)(uintptr_t)m_texQuery.id,
                         ImVec2(m_texQuery.width * s, m_texQuery.height * s));
        } else {
            ImGui::Dummy(ImVec2(panelW, 200.0f));
            ImGui::TextDisabled("(sin imagen)");
        }
        ImGui::EndGroup();
    }

    // Matches strip — full content width
    if (m_texMatches.valid()) {
        ImGui::Spacing();
        ImGui::TextDisabled(
            "Correspondencias  (verde: matches validos, max. 100 mostrados):");
        float maxMatchH = 420.0f;
        float s = std::min(contentW / static_cast<float>(m_texMatches.width),
                           maxMatchH / static_cast<float>(m_texMatches.height));
        ImGui::Image((ImTextureID)(uintptr_t)m_texMatches.id,
                     ImVec2(m_texMatches.width * s, m_texMatches.height * s));
    }

    ImGui::PopID();
}
