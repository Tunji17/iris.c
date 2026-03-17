/*
 * Iris HTTP Server Entry Point
 *
 * Loads a model and starts an HTTP server exposing the Iris API.
 *
 * Usage:
 *   iris-server --dir model/ [--port 8080] [--mmap] [--no-mmap]
 */

#include "iris.h"
#include "iris_http.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#ifdef USE_METAL
#include "iris_metal.h"
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif

#define DEFAULT_PORT 8080

static void print_usage(const char *prog) {
    fprintf(stderr, "Iris HTTP Server\n\n");
    fprintf(stderr, "Usage: %s --dir PATH [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d, --dir PATH        Path to model directory\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -p, --port N          Port to listen on (default: %d)\n", DEFAULT_PORT);
    fprintf(stderr, "  -m, --mmap            Use memory-mapped weights (default)\n");
    fprintf(stderr, "      --no-mmap         Disable mmap, load all weights upfront\n");
    fprintf(stderr, "      --base            Force base model mode\n");
    fprintf(stderr, "  -h, --help            Show this help\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -d flux-klein-4b\n", prog);
    fprintf(stderr, "  %s -d flux-klein-4b --port 9090\n", prog);
    fprintf(stderr, "  %s -d zimage-turbo --no-mmap\n", prog);
}

int main(int argc, char *argv[]) {
#ifdef USE_METAL
    iris_metal_init();
#endif

    static struct option long_options[] = {
        {"dir",      required_argument, 0, 'd'},
        {"port",     required_argument, 0, 'p'},
        {"mmap",     no_argument,       0, 'm'},
        {"no-mmap",  no_argument,       0, 'M'},
        {"base",     no_argument,       0, 'B'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    char *model_dir = NULL;
    int port = DEFAULT_PORT;
    int use_mmap = 1;
    int force_base = 0;

    int opt;
    while ((opt = getopt_long(argc, argv, "d:p:mMBh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd': model_dir = optarg; break;
            case 'p': port = atoi(optarg); break;
            case 'm': use_mmap = 1; break;
            case 'M': use_mmap = 0; break;
            case 'B': force_base = 1; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "Error: Model directory (-d) is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (port < 1 || port > 65535) {
        fprintf(stderr, "Error: Port must be between 1 and 65535\n");
        return 1;
    }

    /* Print backend info */
#ifdef USE_METAL
    fprintf(stderr, "Backend: MPS (Metal GPU)\n");
#elif defined(USE_BLAS)
    fprintf(stderr, "Backend: BLAS\n");
#else
    fprintf(stderr, "Backend: Generic (pure C)\n");
#endif

    /* Load model */
    fprintf(stderr, "Loading model from %s...\n", model_dir);

    iris_ctx *ctx = iris_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to load model: %s\n", iris_get_error());
        return 1;
    }

    if (use_mmap) {
        iris_set_mmap(ctx, 1);
    }

    if (force_base) {
        iris_set_base_mode(ctx);
    }

    fprintf(stderr, "Model: %s\n", iris_model_info(ctx));

    /* Start HTTP server (blocks until shutdown) */
    int rc = iris_http_serve(ctx, port);

    /* Cleanup */
    iris_free(ctx);

#ifdef USE_METAL
    iris_metal_cleanup();
#endif

    return rc;
}
