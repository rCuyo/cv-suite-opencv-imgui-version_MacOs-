#include "app/App.h"
#include <cstdio>

int main()
{
    App app("CV Suite v1.2 — OpenCV 4 + ImGui", 1280, 800);

    if (!app.init()) {
        std::fprintf(stderr, "[CVSuite] Failed to initialise application.\n");
        return -1;
    }

    app.run();
    return 0;
}
