/*
 * Iris HTTP API Server
 *
 * Production HTTP server exposing Iris image generation via REST endpoints.
 * Async job-based: POST endpoints return a job ID, clients poll GET /jobs/{id}.
 *
 * Endpoints:
 *   GET  /health                  - Health check
 *   GET  /info                    - Model information
 *   POST /generate                - Text-to-image generation
 *   POST /img2img                 - Image-to-image (Flux only)
 *   POST /multiref                - Multi-reference generation
 *   POST /encode-text             - Text encoding
 *   POST /generate-with-embeddings - Generate from pre-computed embeddings
 *   POST /vae/encode              - VAE image encoding
 *   POST /vae/decode              - VAE latent decoding
 *   GET  /jobs/{id}               - Poll job status / retrieve result
 *   DELETE /jobs/{id}             - Delete completed job
 */

#ifndef IRIS_HTTP_H
#define IRIS_HTTP_H

#include "iris.h"

/*
 * Start HTTP server. Blocks until SIGINT/SIGTERM.
 * ctx: loaded iris context (reused across requests)
 * port: TCP port to listen on
 * Returns 0 on clean shutdown, -1 on error.
 */
int iris_http_serve(iris_ctx *ctx, int port);

#endif /* IRIS_HTTP_H */
