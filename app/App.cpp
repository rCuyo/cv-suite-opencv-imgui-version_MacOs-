// OpenGL 3.3 core profile — must be included BEFORE GLFW when GLFW_INCLUDE_NONE is set
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "App.h"
#include "modules/ThresholdModule.h"
#include "modules/EdgeDetectionModule.h"

#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────

static void glfwErrorCallback(int error, const char* description)
{
    std::fprintf(stderr, "[GLFW] Error %d: %s\n", error, description);
}

// ─────────────────────────────────────────────────────────────────────────────

App::App(const std::string& title, int width, int height)
    : m_title(title), m_width(width), m_height(height)
{}

App::~App()
{
    shutdown();
}

// ─── init ────────────────────────────────────────────────────────────────────

bool App::init()
{
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        std::fprintf(stderr, "[GLFW] glfwInit() failed.\n");
        return false;
    }

    // Request OpenGL 3.3 Core Profile (required on macOS)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Mandatory on macOS

    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if (!m_window) {
        std::fprintf(stderr, "[GLFW] Failed to create window.\n");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // vsync

    // ── ImGui ────────────────────────────────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    // Tweak style for a cleaner look
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding   = 4.0f;
    style.FrameRounding    = 3.0f;
    style.GrabRounding     = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.ItemSpacing      = ImVec2(8, 6);

    // Backends: GLFW + OpenGL 3 / GLSL 150
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // ── Modules ──────────────────────────────────────────────────────────────
    m_thresholdModule = std::make_unique<ThresholdModule>();
    m_edgeModule      = std::make_unique<EdgeDetectionModule>();

    return true;
}

// ─── Main loop ───────────────────────────────────────────────────────────────

void App::run()
{
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        renderUI();

        ImGui::Render();

        int fbW, fbH;
        glfwGetFramebufferSize(m_window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(m_window);
    }
}

// ─── UI ──────────────────────────────────────────────────────────────────────

void App::renderUI()
{
    // Cover the entire OS window with a single borderless ImGui window
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);

    ImGuiWindowFlags mainFlags =
        ImGuiWindowFlags_NoTitleBar  | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize    | ImGuiWindowFlags_NoMove     |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##Root", nullptr, mainFlags);
    ImGui::PopStyleVar(3);

    // ── Sidebar (fixed 210 px wide) ──────────────────────────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
    ImGui::BeginChild("##Sidebar", ImVec2(210, 0), ImGuiChildFlags_Borders);
    renderSidebar();
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::SameLine();

    // ── Content area (fills remaining space, scrollable) ─────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 10));
    ImGui::BeginChild("##Content", ImVec2(0, 0), ImGuiChildFlags_None,
                      ImGuiWindowFlags_HorizontalScrollbar);
    renderContent();
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::End();
}

void App::renderSidebar()
{
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "CV Suite");
    ImGui::TextDisabled("OpenCV %d.%d", CV_MAJOR_VERSION, CV_MINOR_VERSION);
    ImGui::Separator();
    ImGui::Spacing();

    // Helper: draw a full-width button, highlighted when active
    auto navButton = [&](const char* label, ActiveModule mod) {
        bool active = (m_activeModule == mod);
        if (active)
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::Button(label, ImVec2(-1, 32)))
            m_activeModule = mod;
        if (active)
            ImGui::PopStyleColor();
    };

    ImGui::TextDisabled("MODULES");
    ImGui::Spacing();
    navButton("Thresholding",   ActiveModule::Threshold);
    navButton("Edge Detection", ActiveModule::EdgeDetection);

    // ── Future modules ───────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("COMING SOON");
    ImGui::Spacing();
    ImGui::BeginDisabled();
    ImGui::Button("Histogram",   ImVec2(-1, 32)); // TODO: histogram module
    ImGui::Button("ML Compare",  ImVec2(-1, 32)); // TODO: ML comparison module
    ImGui::EndDisabled();

    // Version footer
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 24.0f);
    ImGui::Separator();
    ImGui::TextDisabled(" v1.0  —  clang / M1");
}

void App::renderContent()
{
    switch (m_activeModule)
    {
    case ActiveModule::None:
        ImGui::Spacing();
        ImGui::SetCursorPosX(
            (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("Select a module from the sidebar.").x) * 0.5f);
        ImGui::TextDisabled("Select a module from the sidebar.");
        break;

    case ActiveModule::Threshold:
        m_thresholdModule->renderUI();
        break;

    case ActiveModule::EdgeDetection:
        m_edgeModule->renderUI();
        break;
    }
}

// ─── Shutdown ────────────────────────────────────────────────────────────────

void App::shutdown()
{
    if (!m_window) return;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();
    m_window = nullptr;
}
