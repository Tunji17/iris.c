# Iris HTTP API Reference

## Overview

The Iris HTTP server exposes image generation capabilities via a REST API.
All generation endpoints are **asynchronous**: POST requests return a job ID,
and clients poll `GET /jobs/{id}` to retrieve results.

**Base URL:** `http://localhost:8080` (default port, configurable with `--port`)

**Content Type:** All request bodies must be `application/json`.

**Image Format:** Images in request/response bodies are base64-encoded PNG or JPEG.

## Starting the Server

```bash
# Build (choose one backend)
make serve-generic   # Pure C, no dependencies
make serve-blas      # BLAS acceleration (~30x faster)
make serve-mps       # Apple Silicon Metal GPU (fastest)

# Run
./iris-server --dir <model-path> [--port 8080] [--mmap] [--no-mmap] [--base]
```

| Flag         | Description                                      | Default |
|--------------|--------------------------------------------------|---------|
| `--dir, -d`  | Path to model directory (required)               | -       |
| `--port, -p` | TCP port to listen on                            | 8080    |
| `--mmap, -m` | Use memory-mapped weights (lower memory)         | on      |
| `--no-mmap`  | Load all weights upfront (faster inference)      | -       |
| `--base`     | Force base model mode (50-step CFG)              | -       |

---

## Async Job Flow

All generation endpoints follow the same pattern:

1. **Submit** a job via POST. Response: `202 Accepted` with a `job_id`.
2. **Poll** via `GET /jobs/{id}`. Returns status: `pending`, `running`, `complete`, or `failed`.
3. **Retrieve** the result when complete (PNG binary or JSON).
4. **Delete** the job via `DELETE /jobs/{id}` to free memory (optional; jobs auto-expire after 5 minutes).

```
POST /generate  -->  {"job_id": 1, "status": "pending"}
GET  /jobs/1    -->  {"job_id": 1, "status": "running"}
GET  /jobs/1    -->  [PNG binary data]  (when complete)
```

---

## Endpoints

### GET /health

Health check. Always returns 200 if the server is running.

**Response:**
```json
{"status": "ok"}
```

---

### GET /info

Returns information about the loaded model.

**Response:**
```json
{
  "model": "flux-klein-4b",
  "is_distilled": true,
  "is_zimage": false,
  "text_dim": 7680,
  "is_non_commercial": false
}
```

| Field              | Type    | Description                                          |
|--------------------|---------|------------------------------------------------------|
| `model`            | string  | Model identifier string                              |
| `is_distilled`     | boolean | `true` for fast distilled models (4 steps default)   |
| `is_zimage`        | boolean | `true` for Z-Image-Turbo, `false` for Flux           |
| `text_dim`         | integer | Text embedding dimension (7680, 12288, or 2560)      |
| `is_non_commercial`| boolean | `true` if model has non-commercial license (e.g. 9B) |

---

### POST /generate

Submit a text-to-image generation job.

**Request body:**
```json
{
  "prompt": "a fluffy orange cat sitting on a windowsill",
  "width": 512,
  "height": 512,
  "steps": 4,
  "seed": 42,
  "guidance": 1.0,
  "schedule": "default",
  "power_alpha": 2.0
}
```

| Field         | Type   | Required | Default          | Description                                    |
|---------------|--------|----------|------------------|------------------------------------------------|
| `prompt`      | string | yes      | -                | Text description of the image to generate      |
| `width`       | int    | no       | 256              | Output width in pixels (64-4096)               |
| `height`      | int    | no       | 256              | Output height in pixels (64-4096)              |
| `steps`       | int    | no       | auto             | Denoising steps (auto: 4 distilled, 50 base, 9 Z-Image) |
| `seed`        | int    | no       | random           | Random seed for reproducibility (-1 = random)  |
| `guidance`    | float  | no       | auto             | CFG guidance scale (auto: 1.0 distilled, 4.0 base, 0.0 Z-Image) |
| `schedule`    | string | no       | `"default"`      | Timestep schedule (see below)                  |
| `power_alpha` | float  | no       | 2.0              | Exponent for power schedule                    |

**Schedule values:**
| Value         | Description                                   |
|---------------|-----------------------------------------------|
| `"default"`   | Model default (sigmoid for Flux, flowmatch for Z-Image) |
| `"linear"`    | Linear timestep schedule                      |
| `"power"`     | Power curve (configurable via `power_alpha`)  |
| `"sigmoid"`   | Flux shifted sigmoid schedule                 |
| `"flowmatch"` | Z-Image FlowMatch Euler schedule              |

**Response:** `202 Accepted`
```json
{"job_id": 1, "status": "pending"}
```

**Result** (via `GET /jobs/1`): PNG image binary, or JSON with `?format=json`:
```json
{
  "job_id": 1,
  "status": "complete",
  "image": "<base64 PNG>",
  "seed": 42,
  "width": 512,
  "height": 512
}
```

---

### POST /img2img

Submit an image-to-image generation job. Uses in-context conditioning
where the reference image is passed as additional tokens to the transformer.

**Note:** Only supported for Flux models, not Z-Image.

**Request body:**
```json
{
  "prompt": "oil painting style",
  "image": "<base64 encoded PNG or JPEG>",
  "width": 512,
  "height": 512,
  "steps": 4,
  "seed": 42,
  "guidance": 1.0,
  "schedule": "default"
}
```

| Field     | Type   | Required | Default | Description                          |
|-----------|--------|----------|---------|--------------------------------------|
| `prompt`  | string | yes      | -       | Text prompt describing the transformation |
| `image`   | string | yes      | -       | Base64-encoded reference image (PNG or JPEG) |
| `width`   | int    | no       | 256     | Output width (64-4096)               |
| `height`  | int    | no       | 256     | Output height (64-4096)              |
| `steps`   | int    | no       | auto    | Denoising steps                      |
| `seed`    | int    | no       | random  | Random seed                          |
| `guidance`| float  | no       | auto    | CFG guidance scale                   |
| `schedule`| string | no       | default | Timestep schedule                    |

**Response:** `202 Accepted`
```json
{"job_id": 2, "status": "pending"}
```

**Result:** Same as `/generate`.

---

### POST /multiref

Submit a multi-reference generation job. Combines multiple reference images
via in-context conditioning with distinct temporal offsets.

**Note:** Only supported for Flux models. Up to 16 reference images.

**Request body:**
```json
{
  "prompt": "combine these into a single scene",
  "images": [
    "<base64 PNG or JPEG>",
    "<base64 PNG or JPEG>"
  ],
  "width": 512,
  "height": 512,
  "steps": 4,
  "seed": 42,
  "guidance": 1.0,
  "schedule": "default"
}
```

| Field     | Type     | Required | Default | Description                            |
|-----------|----------|----------|---------|----------------------------------------|
| `prompt`  | string   | yes      | -       | Text prompt                            |
| `images`  | string[] | yes      | -       | Array of base64-encoded reference images (max 16) |
| `width`   | int      | no       | 256     | Output width (64-4096)                 |
| `height`  | int      | no       | 256     | Output height (64-4096)                |
| `steps`   | int      | no       | auto    | Denoising steps                        |
| `seed`    | int      | no       | random  | Random seed                            |
| `guidance`| float    | no       | auto    | CFG guidance scale                     |
| `schedule`| string   | no       | default | Timestep schedule                      |

**Response:** `202 Accepted`
```json
{"job_id": 3, "status": "pending"}
```

**Result:** Same as `/generate`.

---

### POST /encode-text

Encode a text prompt into embeddings without generating an image.
Useful for caching embeddings or using them with `/generate-with-embeddings`.

**Request body:**
```json
{
  "prompt": "a fluffy cat"
}
```

| Field    | Type   | Required | Description    |
|----------|--------|----------|----------------|
| `prompt` | string | yes      | Text to encode |

**Response:** `202 Accepted`
```json
{"job_id": 4, "status": "pending"}
```

**Result** (via `GET /jobs/4`):
```json
{
  "job_id": 4,
  "status": "complete",
  "embeddings": "<base64 encoded float array>",
  "seq_len": 512,
  "text_dim": 7680
}
```

| Field        | Type   | Description                                     |
|--------------|--------|-------------------------------------------------|
| `embeddings` | string | Base64-encoded float32 array, shape [seq_len, text_dim] |
| `seq_len`    | int    | Number of text tokens (typically 512)           |
| `text_dim`   | int    | Embedding dimension per token                   |

---

### POST /generate-with-embeddings

Generate an image from pre-computed text embeddings.
Use with embeddings from `/encode-text` to skip text encoding on repeated prompts.

**Request body:**
```json
{
  "embeddings": "<base64 encoded float array>",
  "seq_len": 512,
  "width": 512,
  "height": 512,
  "steps": 4,
  "seed": 42,
  "guidance": 1.0,
  "schedule": "default"
}
```

| Field        | Type   | Required | Default | Description                              |
|--------------|--------|----------|---------|------------------------------------------|
| `embeddings` | string | yes      | -       | Base64-encoded float32 embedding array   |
| `seq_len`    | int    | yes      | -       | Number of tokens in the embedding        |
| `width`      | int    | no       | 256     | Output width (64-4096)                   |
| `height`     | int    | no       | 256     | Output height (64-4096)                  |
| `steps`      | int    | no       | auto    | Denoising steps                          |
| `seed`       | int    | no       | random  | Random seed                              |
| `guidance`   | float  | no       | auto    | CFG guidance scale                       |
| `schedule`   | string | no       | default | Timestep schedule                        |

**Response:** `202 Accepted`
```json
{"job_id": 5, "status": "pending"}
```

**Result:** Same as `/generate`.

---

### POST /vae/encode

Encode an image into the VAE latent space. Returns the latent representation.

**Request body:**
```json
{
  "image": "<base64 encoded PNG or JPEG>"
}
```

| Field   | Type   | Required | Description                          |
|---------|--------|----------|--------------------------------------|
| `image` | string | yes      | Base64-encoded image (PNG or JPEG)   |

**Response:** `202 Accepted`
```json
{"job_id": 6, "status": "pending"}
```

**Result** (via `GET /jobs/6`):
```json
{
  "job_id": 6,
  "status": "complete",
  "latent": "<base64 encoded float array>",
  "latent_h": 16,
  "latent_w": 16
}
```

| Field      | Type   | Description                                           |
|------------|--------|-------------------------------------------------------|
| `latent`   | string | Base64-encoded float32 array, shape [128, latent_h, latent_w] |
| `latent_h` | int    | Latent height (input height / 16)                     |
| `latent_w` | int    | Latent width (input width / 16)                       |

---

### POST /vae/decode

Decode a VAE latent back into an image.

**Request body:**
```json
{
  "latent": "<base64 encoded float array>",
  "latent_h": 16,
  "latent_w": 16
}
```

| Field      | Type   | Required | Description                                |
|------------|--------|----------|--------------------------------------------|
| `latent`   | string | yes      | Base64-encoded float32 latent array        |
| `latent_h` | int    | yes      | Latent height                              |
| `latent_w` | int    | yes      | Latent width                               |

**Response:** `202 Accepted`
```json
{"job_id": 7, "status": "pending"}
```

**Result:** PNG image (same as `/generate`).

---

### GET /jobs/{id}

Poll the status of a submitted job and retrieve results.

**Path parameter:** `id` - integer job ID returned by a POST endpoint.

**Query parameters:**

| Parameter | Value  | Description                                |
|-----------|--------|--------------------------------------------|
| `format`  | `json` | Return image results as JSON with base64 instead of raw PNG |

**Response (pending/running):** `200 OK`
```json
{"job_id": 1, "status": "running"}
```

**Response (complete, image result, default):** `200 OK`
- Content-Type: `image/png`
- Body: raw PNG binary data

**Response (complete, image result, `?format=json`):** `200 OK`
```json
{
  "job_id": 1,
  "status": "complete",
  "image": "<base64 PNG>",
  "seed": 42,
  "width": 512,
  "height": 512
}
```

**Response (complete, non-image result):** `200 OK`
- Returns the result JSON directly (e.g. embeddings, latent data)

**Response (failed):** `200 OK`
```json
{"job_id": 1, "status": "failed", "error": "Out of memory"}
```

**Response (not found):** `404 Not Found`
```json
{"error": "Job not found"}
```

---

### DELETE /jobs/{id}

Delete a completed or failed job to free its memory immediately.
Jobs auto-expire after 5 minutes, so this is optional.

Cannot delete pending or running jobs.

**Response (success):** `200 OK`
```json
{"deleted": true}
```

**Response (active job):** `400 Bad Request`
```json
{"error": "Cannot delete active job"}
```

**Response (not found):** `404 Not Found`
```json
{"error": "Job not found"}
```

---

## Error Responses

All errors return JSON with an `error` field:

```json
{"error": "description of the problem"}
```

| Status | Meaning              | Common causes                              |
|--------|----------------------|--------------------------------------------|
| 400    | Bad Request          | Missing required fields, invalid parameters, malformed JSON |
| 404    | Not Found            | Unknown endpoint or expired/unknown job ID |
| 405    | Method Not Allowed   | Wrong HTTP method for the endpoint         |
| 413    | Payload Too Large    | Request body exceeds 50MB limit            |
| 500    | Internal Server Error| Generation failed (model error)            |
| 503    | Service Unavailable  | Job queue is full (64 job limit)           |

---

## CORS

All responses include CORS headers allowing cross-origin requests:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

OPTIONS requests return `204 No Content` for preflight handling.

---

## Concurrency

The server processes **one job at a time** on a dedicated worker thread.
Additional jobs are queued (up to 64) and processed in order.
HTTP requests (job submission, status polling) are handled immediately
on the main thread regardless of whether a job is running.

---

## Examples

### Generate an image

```bash
# Submit
curl -s -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat sitting on a rainbow","width":512,"height":512,"seed":42}'
# {"job_id":1,"status":"pending"}

# Poll until done
curl -s http://localhost:8080/jobs/1
# {"job_id":1,"status":"running"}

# Download result
curl -s http://localhost:8080/jobs/1 -o cat.png
```

### Image-to-image

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w0 photo.png)

# Submit
curl -s -X POST http://localhost:8080/img2img \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"oil painting style\",\"image\":\"$IMAGE_B64\",\"width\":512,\"height\":512}"

# Poll and download
curl -s http://localhost:8080/jobs/1 -o painting.png
```

### Cache and reuse embeddings

```bash
# Encode text once
curl -s -X POST http://localhost:8080/encode-text \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a beautiful sunset over mountains"}'
# {"job_id":1,"status":"pending"}

# Get embeddings
RESULT=$(curl -s "http://localhost:8080/jobs/1?format=json")
EMB=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['embeddings'])")

# Generate multiple images with same embeddings
for SEED in 1 2 3 4; do
  curl -s -X POST http://localhost:8080/generate-with-embeddings \
    -H "Content-Type: application/json" \
    -d "{\"embeddings\":\"$EMB\",\"seq_len\":512,\"width\":512,\"height\":512,\"seed\":$SEED}"
done
```

### Poll loop in bash

```bash
JOB_ID=1
while true; do
  STATUS=$(curl -s http://localhost:8080/jobs/$JOB_ID)
  STATE=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
  if [ "$STATE" = "complete" ]; then
    curl -s http://localhost:8080/jobs/$JOB_ID -o result.png
    echo "Done: result.png"
    break
  elif [ "$STATE" = "failed" ]; then
    echo "Failed: $STATUS"
    break
  fi
  sleep 1
done
```
