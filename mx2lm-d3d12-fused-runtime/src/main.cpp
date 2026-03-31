#include "runtime/simulation.h"
#include "runtime/config.h"
#include <cstdio>
#include <cstring>

using namespace mx2lm;

static void PrintUsage(const char* exe) {
    printf("Usage: %s [options]\n", exe);
    printf("  --entities N       Number of entities (default: 4096)\n");
    printf("  --steps N          Max steps, 0=unlimited (default: 0)\n");
    printf("  --scxq2 PATH       Load initial state from SCXQ2 file\n");
    printf("  --experts N        Number of MoE experts (default: 8)\n");
    printf("  --readback         Enable debug readback\n");
    printf("  --debug-layer      Enable D3D12 debug layer\n");
    printf("  --help             Print this help\n");
}

int main(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            PrintUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--entities") == 0 && i + 1 < argc) {
            cfg.entityCount = (uint32_t)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            cfg.maxSteps = (uint32_t)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--scxq2") == 0 && i + 1 < argc) {
            cfg.scxq2Path = argv[++i];
        }
        else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            cfg.numExperts = (uint32_t)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--readback") == 0) {
            cfg.enableReadback = true;
        }
        else if (strcmp(argv[i], "--debug-layer") == 0) {
            cfg.enableDebugLayer = true;
        }
        else {
            printf("[warn] Unknown argument: %s\n", argv[i]);
        }
    }

    printf("==============================================\n");
    printf("  MX2LM D3D12 Fused Runtime\n");
    printf("  entities=%u  experts=%u  steps=%u\n",
           cfg.entityCount, cfg.numExperts, cfg.maxSteps);
    printf("==============================================\n\n");

    try {
        Simulation sim;
        sim.Init(cfg);
        sim.Run();
        sim.Shutdown();
    }
    catch (const std::exception& e) {
        fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    }

    return 0;
}
