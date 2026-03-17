/*
 * Iris HTTP API Server
 *
 * Production HTTP server for Iris image generation.
 * Pure C implementation using POSIX sockets, no external dependencies.
 *
 * Architecture:
 *   Main thread: accept() loop, parse HTTP, enqueue jobs, return status
 *   Worker thread: dequeue jobs, run iris_* calls, store results
 *
 * All iris_* calls are serialized on the worker thread to avoid
 * thread-safety issues with global state (error buffer, callbacks, GPU).
 */

#include "iris_http.h"
#include "iris.h"
#include "iris_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define HTTP_MAX_REQUEST_SIZE   (50 * 1024 * 1024)  /* 50MB max request body */
#define HTTP_MAX_HEADERS       32
#define HTTP_READ_BUF_SIZE     8192
#define HTTP_MAX_PATH          512
#define HTTP_MAX_QUERY         512

#define JOB_QUEUE_SIZE         64
#define JOB_EXPIRY_SECONDS     300   /* 5 minutes */
#define MAX_REFS               16

/* ========================================================================
 * Base64 Encode / Decode
 * ======================================================================== */

static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static char *b64_encode(const unsigned char *data, size_t len, size_t *out_len) {
    size_t encoded_len = 4 * ((len + 2) / 3);
    char *encoded = malloc(encoded_len + 1);
    if (!encoded) return NULL;

    size_t i, j;
    for (i = 0, j = 0; i < len; ) {
        uint32_t a = i < len ? data[i++] : 0;
        uint32_t b = i < len ? data[i++] : 0;
        uint32_t c = i < len ? data[i++] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;

        encoded[j++] = b64_table[(triple >> 18) & 0x3F];
        encoded[j++] = b64_table[(triple >> 12) & 0x3F];
        encoded[j++] = b64_table[(triple >> 6) & 0x3F];
        encoded[j++] = b64_table[triple & 0x3F];
    }

    int pad = len % 3;
    if (pad) {
        encoded[encoded_len - 1] = '=';
        if (pad == 1) encoded[encoded_len - 2] = '=';
    }

    encoded[encoded_len] = '\0';
    if (out_len) *out_len = encoded_len;
    return encoded;
}

static const int b64_decode_table[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};

static uint8_t *b64_decode(const char *src, size_t src_len, size_t *out_len) {
    if (!src || src_len == 0) return NULL;

    /* Skip trailing whitespace/padding for length calculation */
    while (src_len > 0 && (src[src_len - 1] == '=' || src[src_len - 1] == '\n'
                           || src[src_len - 1] == '\r' || src[src_len - 1] == ' '))
        src_len--;

    size_t decoded_len = (src_len * 3) / 4;
    uint8_t *decoded = malloc(decoded_len + 1);
    if (!decoded) return NULL;

    size_t i = 0, j = 0;
    while (i < src_len) {
        int a = (i < src_len) ? b64_decode_table[(unsigned char)src[i++]] : -2;
        int b = (i < src_len) ? b64_decode_table[(unsigned char)src[i++]] : -2;
        int c = (i < src_len) ? b64_decode_table[(unsigned char)src[i++]] : -2;
        int d = (i < src_len) ? b64_decode_table[(unsigned char)src[i++]] : -2;

        /* Skip whitespace */
        if (a == -1 || b == -1) continue;

        if (a < 0) a = 0;
        if (b < 0) b = 0;
        if (c < 0) c = 0;
        if (d < 0) d = 0;

        uint32_t triple = ((uint32_t)a << 18) | ((uint32_t)b << 12)
                         | ((uint32_t)c << 6) | (uint32_t)d;

        if (j < decoded_len) decoded[j++] = (triple >> 16) & 0xFF;
        if (j < decoded_len) decoded[j++] = (triple >> 8) & 0xFF;
        if (j < decoded_len) decoded[j++] = triple & 0xFF;
    }

    *out_len = j;
    return decoded;
}

/* ========================================================================
 * Minimal JSON Parser
 *
 * Supports flat objects with string, number, boolean, null values,
 * and arrays of strings. Enough for our API request bodies.
 * ======================================================================== */

#define JSON_MAX_KEYS    32
#define JSON_MAX_STRINGS 16  /* max array elements */

typedef enum {
    JSON_STRING,
    JSON_NUMBER,
    JSON_BOOL,
    JSON_NULL,
    JSON_ARRAY_STRINGS
} json_type_t;

typedef struct {
    char *key;
    json_type_t type;
    union {
        char *str;
        double num;
        int boolean;
        struct {
            char **items;
            int count;
        } arr;
    } val;
} json_kv_t;

typedef struct {
    json_kv_t entries[JSON_MAX_KEYS];
    int count;
} json_obj_t;

/* Skip whitespace */
static const char *json_skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Parse a JSON string, return malloc'd copy. Advances *pp past closing quote. */
static char *json_parse_string(const char **pp, const char *end) {
    const char *p = *pp;
    if (p >= end || *p != '"') return NULL;
    p++;

    /* Find end of string (handle escape sequences) */
    const char *start = p;
    size_t len = 0;
    while (p < end && *p != '"') {
        if (*p == '\\') { p++; if (p < end) p++; }
        else p++;
        len++;
    }

    /* Allocate and copy with basic unescape */
    char *result = malloc(len + 1);
    if (!result) return NULL;

    p = start;
    size_t j = 0;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            p++;
            if (p < end) {
                switch (*p) {
                    case '"':  result[j++] = '"'; break;
                    case '\\': result[j++] = '\\'; break;
                    case '/':  result[j++] = '/'; break;
                    case 'n':  result[j++] = '\n'; break;
                    case 'r':  result[j++] = '\r'; break;
                    case 't':  result[j++] = '\t'; break;
                    default:   result[j++] = *p; break;
                }
                p++;
            }
        } else {
            result[j++] = *p++;
        }
    }
    result[j] = '\0';

    if (p < end) p++; /* skip closing quote */
    *pp = p;
    return result;
}

/* Parse a JSON number */
static double json_parse_number(const char **pp, const char *end) {
    const char *p = *pp;
    char buf[64];
    int i = 0;
    while (p < end && i < 63 && ((*p >= '0' && *p <= '9') || *p == '-'
           || *p == '+' || *p == '.' || *p == 'e' || *p == 'E')) {
        buf[i++] = *p++;
    }
    buf[i] = '\0';
    *pp = p;
    return atof(buf);
}

/* Parse a flat JSON object */
static json_obj_t *json_parse(const char *body, size_t body_len) {
    json_obj_t *obj = calloc(1, sizeof(json_obj_t));
    if (!obj) return NULL;

    const char *p = body;
    const char *end = body + body_len;

    p = json_skip_ws(p, end);
    if (p >= end || *p != '{') { free(obj); return NULL; }
    p++;

    while (p < end && obj->count < JSON_MAX_KEYS) {
        p = json_skip_ws(p, end);
        if (p >= end || *p == '}') break;

        /* Skip comma */
        if (*p == ',') { p++; p = json_skip_ws(p, end); }
        if (p >= end || *p == '}') break;

        /* Parse key */
        char *key = json_parse_string(&p, end);
        if (!key) break;

        p = json_skip_ws(p, end);
        if (p >= end || *p != ':') { free(key); break; }
        p++;
        p = json_skip_ws(p, end);

        json_kv_t *kv = &obj->entries[obj->count];
        kv->key = key;

        /* Parse value */
        if (*p == '"') {
            kv->type = JSON_STRING;
            kv->val.str = json_parse_string(&p, end);
        } else if (*p == '[') {
            /* Array of strings */
            kv->type = JSON_ARRAY_STRINGS;
            kv->val.arr.items = calloc(JSON_MAX_STRINGS, sizeof(char *));
            kv->val.arr.count = 0;
            p++;
            while (p < end && kv->val.arr.count < JSON_MAX_STRINGS) {
                p = json_skip_ws(p, end);
                if (p >= end || *p == ']') break;
                if (*p == ',') { p++; continue; }
                if (*p == '"') {
                    kv->val.arr.items[kv->val.arr.count++] = json_parse_string(&p, end);
                } else {
                    break;
                }
            }
            if (p < end && *p == ']') p++;
        } else if (*p == 't' || *p == 'f') {
            kv->type = JSON_BOOL;
            if (strncmp(p, "true", 4) == 0) { kv->val.boolean = 1; p += 4; }
            else if (strncmp(p, "false", 5) == 0) { kv->val.boolean = 0; p += 5; }
        } else if (*p == 'n') {
            kv->type = JSON_NULL;
            if (strncmp(p, "null", 4) == 0) p += 4;
        } else {
            kv->type = JSON_NUMBER;
            kv->val.num = json_parse_number(&p, end);
        }

        obj->count++;
    }

    return obj;
}

static const char *json_get_string(const json_obj_t *obj, const char *key) {
    for (int i = 0; i < obj->count; i++) {
        if (obj->entries[i].type == JSON_STRING && strcmp(obj->entries[i].key, key) == 0)
            return obj->entries[i].val.str;
    }
    return NULL;
}

static int json_get_int(const json_obj_t *obj, const char *key, int def) {
    for (int i = 0; i < obj->count; i++) {
        if (obj->entries[i].type == JSON_NUMBER && strcmp(obj->entries[i].key, key) == 0)
            return (int)obj->entries[i].val.num;
    }
    return def;
}

static double json_get_float(const json_obj_t *obj, const char *key, double def) {
    for (int i = 0; i < obj->count; i++) {
        if (obj->entries[i].type == JSON_NUMBER && strcmp(obj->entries[i].key, key) == 0)
            return obj->entries[i].val.num;
    }
    return def;
}

static int json_get_string_array(const json_obj_t *obj, const char *key,
                                 char ***items, int *count) {
    for (int i = 0; i < obj->count; i++) {
        if (obj->entries[i].type == JSON_ARRAY_STRINGS
            && strcmp(obj->entries[i].key, key) == 0) {
            *items = obj->entries[i].val.arr.items;
            *count = obj->entries[i].val.arr.count;
            return 1;
        }
    }
    return 0;
}

static void json_free(json_obj_t *obj) {
    if (!obj) return;
    for (int i = 0; i < obj->count; i++) {
        free(obj->entries[i].key);
        if (obj->entries[i].type == JSON_STRING) {
            free(obj->entries[i].val.str);
        } else if (obj->entries[i].type == JSON_ARRAY_STRINGS) {
            for (int j = 0; j < obj->entries[i].val.arr.count; j++)
                free(obj->entries[i].val.arr.items[j]);
            free(obj->entries[i].val.arr.items);
        }
    }
    free(obj);
}

/* ========================================================================
 * HTTP Request Parser
 * ======================================================================== */

typedef struct {
    char method[16];
    char path[HTTP_MAX_PATH];
    char query[HTTP_MAX_QUERY];
    char *body;
    size_t body_len;
    size_t content_length;
    char content_type[128];
} http_request_t;

/* Read a complete HTTP request from the socket */
static int http_read_request(int fd, http_request_t *req) {
    memset(req, 0, sizeof(*req));

    /* Read headers */
    char header_buf[HTTP_READ_BUF_SIZE];
    size_t header_len = 0;
    int found_end = 0;

    while (header_len < sizeof(header_buf) - 1) {
        ssize_t n = recv(fd, header_buf + header_len, sizeof(header_buf) - 1 - header_len, 0);
        if (n <= 0) return -1;
        header_len += n;
        header_buf[header_len] = '\0';

        if (strstr(header_buf, "\r\n\r\n")) {
            found_end = 1;
            break;
        }
    }
    if (!found_end) return -1;

    /* Parse request line */
    char *line_end = strstr(header_buf, "\r\n");
    if (!line_end) return -1;

    /* METHOD /path?query HTTP/1.x */
    char *p = header_buf;
    char *method_end = strchr(p, ' ');
    if (!method_end) return -1;
    size_t mlen = method_end - p;
    if (mlen >= sizeof(req->method)) mlen = sizeof(req->method) - 1;
    memcpy(req->method, p, mlen);
    req->method[mlen] = '\0';

    p = method_end + 1;
    char *path_end = strchr(p, ' ');
    if (!path_end) return -1;

    /* Split path and query */
    char *qmark = memchr(p, '?', path_end - p);
    if (qmark) {
        size_t plen = qmark - p;
        if (plen >= sizeof(req->path)) plen = sizeof(req->path) - 1;
        memcpy(req->path, p, plen);
        req->path[plen] = '\0';

        size_t qlen = path_end - qmark - 1;
        if (qlen >= sizeof(req->query)) qlen = sizeof(req->query) - 1;
        memcpy(req->query, qmark + 1, qlen);
        req->query[qlen] = '\0';
    } else {
        size_t plen = path_end - p;
        if (plen >= sizeof(req->path)) plen = sizeof(req->path) - 1;
        memcpy(req->path, p, plen);
        req->path[plen] = '\0';
    }

    /* Parse headers for Content-Length and Content-Type */
    char *header_start = line_end + 2;
    char *body_start = strstr(header_buf, "\r\n\r\n") + 4;

    char *h = header_start;
    while (h < body_start - 2) {
        char *h_end = strstr(h, "\r\n");
        if (!h_end) break;

        if (strncasecmp(h, "Content-Length:", 15) == 0) {
            req->content_length = (size_t)atoll(h + 15);
        } else if (strncasecmp(h, "Content-Type:", 13) == 0) {
            const char *val = h + 13;
            while (*val == ' ') val++;
            size_t vlen = h_end - val;
            if (vlen >= sizeof(req->content_type)) vlen = sizeof(req->content_type) - 1;
            memcpy(req->content_type, val, vlen);
            req->content_type[vlen] = '\0';
        }

        h = h_end + 2;
    }

    /* Read body if present */
    if (req->content_length > 0) {
        if (req->content_length > HTTP_MAX_REQUEST_SIZE) return -1;

        req->body = malloc(req->content_length + 1);
        if (!req->body) return -1;

        /* Copy any body data already read with headers */
        size_t already_read = header_len - (body_start - header_buf);
        if (already_read > req->content_length) already_read = req->content_length;
        memcpy(req->body, body_start, already_read);
        req->body_len = already_read;

        /* Read remaining body */
        while (req->body_len < req->content_length) {
            ssize_t n = recv(fd, req->body + req->body_len,
                             req->content_length - req->body_len, 0);
            if (n <= 0) { free(req->body); req->body = NULL; return -1; }
            req->body_len += n;
        }
        req->body[req->body_len] = '\0';
    }

    return 0;
}

static void http_request_free(http_request_t *req) {
    free(req->body);
    req->body = NULL;
}

/* Check if query string contains a key=value pair */
static int query_has(const char *query, const char *key, const char *value) {
    char needle[128];
    snprintf(needle, sizeof(needle), "%s=%s", key, value);
    return strstr(query, needle) != NULL;
}

/* ========================================================================
 * HTTP Response Helpers
 * ======================================================================== */

static void http_respond_raw(int fd, int status, const char *status_text,
                             const char *content_type, const void *body, size_t body_len) {
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);

    send(fd, header, hlen, MSG_NOSIGNAL);
    if (body && body_len > 0) {
        send(fd, body, body_len, MSG_NOSIGNAL);
    }
}

static void http_respond_json(int fd, int status, const char *status_text,
                              const char *json) {
    http_respond_raw(fd, status, status_text, "application/json",
                     json, strlen(json));
}

static void http_respond_png(int fd, const uint8_t *png_data, size_t png_len) {
    http_respond_raw(fd, 200, "OK", "image/png", png_data, png_len);
}

static void http_respond_error(int fd, int status, const char *message) {
    char json[512];
    snprintf(json, sizeof(json), "{\"error\":\"%s\"}", message);
    const char *status_text = "Error";
    switch (status) {
        case 400: status_text = "Bad Request"; break;
        case 404: status_text = "Not Found"; break;
        case 405: status_text = "Method Not Allowed"; break;
        case 413: status_text = "Payload Too Large"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 503: status_text = "Service Unavailable"; break;
    }
    http_respond_json(fd, status, status_text, json);
}

/* ========================================================================
 * Image Helpers
 * ======================================================================== */

/* Decode a base64-encoded image (PNG or JPEG) into an iris_image */
static iris_image *decode_b64_image(const char *b64, size_t b64_len) {
    size_t raw_len;
    uint8_t *raw = b64_decode(b64, b64_len, &raw_len);
    if (!raw) return NULL;

    iris_image *img = iris_image_load_mem(raw, raw_len);
    free(raw);
    return img;
}

/* ========================================================================
 * Job Queue
 * ======================================================================== */

typedef enum {
    JOB_PENDING,
    JOB_RUNNING,
    JOB_COMPLETE,
    JOB_FAILED
} job_state_t;

typedef enum {
    JOB_GENERATE,
    JOB_IMG2IMG,
    JOB_MULTIREF,
    JOB_ENCODE_TEXT,
    JOB_GENERATE_WITH_EMBEDDINGS,
    JOB_VAE_ENCODE,
    JOB_VAE_DECODE
} job_type_t;

typedef struct {
    int id;
    job_state_t state;
    job_type_t type;

    /* Input */
    char *prompt;
    iris_params params;
    int64_t seed;

    /* For img2img / vae/encode: single input image */
    uint8_t *input_image_data;
    size_t input_image_len;

    /* For multiref: array of input images */
    uint8_t **ref_images_data;
    size_t *ref_images_lens;
    int num_refs;

    /* For generate-with-embeddings */
    float *embeddings;
    int emb_seq_len;

    /* For vae/decode */
    float *latent;
    int latent_h, latent_w;

    /* Output */
    uint8_t *result_png;
    size_t result_png_len;
    char *result_json;
    int result_width, result_height;
    int64_t result_seed;
    char error_msg[256];

    /* Lifecycle */
    time_t created_at;
    time_t completed_at;
    int active;  /* slot in use */
} http_job_t;

typedef struct {
    http_job_t jobs[JOB_QUEUE_SIZE];
    int next_id;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int shutdown;
} job_queue_t;

static void job_queue_init(job_queue_t *q) {
    memset(q, 0, sizeof(*q));
    q->next_id = 1;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->cond, NULL);
}

static void job_queue_destroy(job_queue_t *q) {
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond);

    for (int i = 0; i < JOB_QUEUE_SIZE; i++) {
        http_job_t *j = &q->jobs[i];
        if (!j->active) continue;
        free(j->prompt);
        free(j->input_image_data);
        free(j->result_png);
        free(j->result_json);
        free(j->embeddings);
        free(j->latent);
        if (j->ref_images_data) {
            for (int k = 0; k < j->num_refs; k++) free(j->ref_images_data[k]);
            free(j->ref_images_data);
            free(j->ref_images_lens);
        }
    }
}

/* Clean up expired completed jobs */
static void job_queue_expire(job_queue_t *q) {
    time_t now = time(NULL);
    for (int i = 0; i < JOB_QUEUE_SIZE; i++) {
        http_job_t *j = &q->jobs[i];
        if (j->active && (j->state == JOB_COMPLETE || j->state == JOB_FAILED)
            && (now - j->completed_at) > JOB_EXPIRY_SECONDS) {
            free(j->prompt); j->prompt = NULL;
            free(j->input_image_data); j->input_image_data = NULL;
            free(j->result_png); j->result_png = NULL;
            free(j->result_json); j->result_json = NULL;
            free(j->embeddings); j->embeddings = NULL;
            free(j->latent); j->latent = NULL;
            if (j->ref_images_data) {
                for (int k = 0; k < j->num_refs; k++) free(j->ref_images_data[k]);
                free(j->ref_images_data); j->ref_images_data = NULL;
                free(j->ref_images_lens); j->ref_images_lens = NULL;
            }
            j->active = 0;
        }
    }
}

/* Enqueue a new job. Returns job ID, or -1 if queue is full. */
static int job_enqueue(job_queue_t *q, http_job_t *template) {
    pthread_mutex_lock(&q->mutex);

    job_queue_expire(q);

    /* Find free slot */
    int slot = -1;
    for (int i = 0; i < JOB_QUEUE_SIZE; i++) {
        if (!q->jobs[i].active) { slot = i; break; }
    }

    if (slot < 0) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }

    http_job_t *j = &q->jobs[slot];
    *j = *template;
    j->id = q->next_id++;
    j->state = JOB_PENDING;
    j->active = 1;
    j->created_at = time(NULL);

    int id = j->id;

    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
    return id;
}

/* Find a pending job and mark it as running. Returns slot index, or -1. */
static int job_dequeue(job_queue_t *q) {
    for (int i = 0; i < JOB_QUEUE_SIZE; i++) {
        if (q->jobs[i].active && q->jobs[i].state == JOB_PENDING) {
            q->jobs[i].state = JOB_RUNNING;
            return i;
        }
    }
    return -1;
}

/* Find a job by ID. Must hold mutex. Returns slot index or -1. */
static int job_find(job_queue_t *q, int id) {
    for (int i = 0; i < JOB_QUEUE_SIZE; i++) {
        if (q->jobs[i].active && q->jobs[i].id == id) return i;
    }
    return -1;
}

/* ========================================================================
 * Parse Schedule Name
 * ======================================================================== */

static int parse_schedule(const char *name) {
    if (!name) return IRIS_SCHEDULE_DEFAULT;
    if (strcmp(name, "linear") == 0) return IRIS_SCHEDULE_LINEAR;
    if (strcmp(name, "power") == 0) return IRIS_SCHEDULE_POWER;
    if (strcmp(name, "sigmoid") == 0) return IRIS_SCHEDULE_SIGMOID;
    if (strcmp(name, "flowmatch") == 0) return IRIS_SCHEDULE_FLOWMATCH;
    return IRIS_SCHEDULE_DEFAULT;
}

/* ========================================================================
 * Build iris_params from JSON
 * ======================================================================== */

static iris_params params_from_json(const json_obj_t *obj) {
    iris_params p = IRIS_PARAMS_DEFAULT;
    p.width = json_get_int(obj, "width", IRIS_DEFAULT_WIDTH);
    p.height = json_get_int(obj, "height", IRIS_DEFAULT_HEIGHT);
    p.num_steps = json_get_int(obj, "steps", 0);
    p.seed = (int64_t)json_get_float(obj, "seed", -1.0);
    p.guidance = (float)json_get_float(obj, "guidance", 0.0);
    p.schedule = parse_schedule(json_get_string(obj, "schedule"));
    p.power_alpha = (float)json_get_float(obj, "power_alpha", 2.0);
    return p;
}

/* Validate params. Returns NULL if ok, or error message string. */
static const char *validate_params(const iris_params *p) {
    if (p->width < 64 || p->width > 4096) return "width must be between 64 and 4096";
    if (p->height < 64 || p->height > 4096) return "height must be between 64 and 4096";
    if (p->num_steps < 0 || p->num_steps > IRIS_MAX_STEPS)
        return "steps must be between 0 and 256";
    return NULL;
}

/* ========================================================================
 * Worker Thread
 *
 * Processes jobs from the queue, calling iris_* functions.
 * Only this thread touches iris_ctx, ensuring thread safety.
 * ======================================================================== */

typedef struct {
    iris_ctx *ctx;
    job_queue_t *queue;
} worker_ctx_t;

static void worker_process_job(iris_ctx *ctx, http_job_t *job) {
    /* Set seed */
    int64_t seed = job->seed;
    if (seed < 0) seed = (int64_t)time(NULL) ^ (int64_t)(uintptr_t)job;
    iris_set_seed(seed);
    job->result_seed = seed;

    /* Resolve auto-parameters */
    if (job->params.num_steps <= 0) {
        if (iris_is_zimage(ctx))
            job->params.num_steps = 9;
        else
            job->params.num_steps = iris_is_distilled(ctx) ? 4 : 50;
    }
    if (job->params.guidance <= 0) {
        if (iris_is_zimage(ctx))
            job->params.guidance = 0.0f;
        else
            job->params.guidance = iris_is_distilled(ctx) ? 1.0f : 4.0f;
    }

    switch (job->type) {
    case JOB_GENERATE: {
        iris_image *img = iris_generate(ctx, job->prompt, &job->params);
        if (!img) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "%s", iris_get_error());
            return;
        }
        job->result_width = img->width;
        job->result_height = img->height;
        job->result_png = iris_image_to_png_mem(img, &job->result_png_len, seed);
        iris_image_free(img);
        if (!job->result_png) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to encode PNG");
            return;
        }
        job->state = JOB_COMPLETE;
        break;
    }

    case JOB_IMG2IMG: {
        iris_image *input = decode_b64_image((const char *)job->input_image_data,
                                             job->input_image_len);
        if (!input) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to decode input image");
            return;
        }
        iris_image *img = iris_img2img(ctx, job->prompt, input, &job->params);
        iris_image_free(input);
        if (!img) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "%s", iris_get_error());
            return;
        }
        job->result_width = img->width;
        job->result_height = img->height;
        job->result_png = iris_image_to_png_mem(img, &job->result_png_len, seed);
        iris_image_free(img);
        if (!job->result_png) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to encode PNG");
            return;
        }
        job->state = JOB_COMPLETE;
        break;
    }

    case JOB_MULTIREF: {
        iris_image *refs[MAX_REFS];
        int num = job->num_refs;
        if (num > MAX_REFS) num = MAX_REFS;

        for (int i = 0; i < num; i++) {
            refs[i] = decode_b64_image((const char *)job->ref_images_data[i],
                                       job->ref_images_lens[i]);
            if (!refs[i]) {
                for (int k = 0; k < i; k++) iris_image_free(refs[k]);
                job->state = JOB_FAILED;
                snprintf(job->error_msg, sizeof(job->error_msg),
                         "Failed to decode reference image %d", i + 1);
                return;
            }
        }

        iris_image *img = iris_multiref(ctx, job->prompt,
                                        (const iris_image **)refs, num, &job->params);
        for (int i = 0; i < num; i++) iris_image_free(refs[i]);

        if (!img) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "%s", iris_get_error());
            return;
        }
        job->result_width = img->width;
        job->result_height = img->height;
        job->result_png = iris_image_to_png_mem(img, &job->result_png_len, seed);
        iris_image_free(img);
        if (!job->result_png) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to encode PNG");
            return;
        }
        job->state = JOB_COMPLETE;
        break;
    }

    case JOB_ENCODE_TEXT: {
        int seq_len = 0;
        float *emb = iris_encode_text(ctx, job->prompt, &seq_len);
        if (!emb) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "%s", iris_get_error());
            return;
        }
        int text_dim = iris_text_dim(ctx);
        size_t emb_bytes = (size_t)seq_len * text_dim * sizeof(float);
        size_t b64_len;
        char *b64 = b64_encode((const unsigned char *)emb, emb_bytes, &b64_len);
        free(emb);

        if (!b64) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to encode embeddings");
            return;
        }

        /* Build JSON result */
        size_t json_len = b64_len + 256;
        job->result_json = malloc(json_len);
        snprintf(job->result_json, json_len,
                 "{\"embeddings\":\"%s\",\"seq_len\":%d,\"text_dim\":%d}",
                 b64, seq_len, text_dim);
        free(b64);
        job->state = JOB_COMPLETE;
        break;
    }

    case JOB_GENERATE_WITH_EMBEDDINGS: {
        iris_image *img = iris_generate_with_embeddings(ctx, job->embeddings,
                                                        job->emb_seq_len, &job->params);
        if (!img) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "%s", iris_get_error());
            return;
        }
        job->result_width = img->width;
        job->result_height = img->height;
        job->result_png = iris_image_to_png_mem(img, &job->result_png_len, seed);
        iris_image_free(img);
        if (!job->result_png) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to encode PNG");
            return;
        }
        job->state = JOB_COMPLETE;
        break;
    }

    case JOB_VAE_ENCODE: {
        iris_image *input = decode_b64_image((const char *)job->input_image_data,
                                             job->input_image_len);
        if (!input) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to decode input image");
            return;
        }
        int out_h = 0, out_w = 0;
        float *latent = iris_encode_image(ctx, input, &out_h, &out_w);
        iris_image_free(input);

        if (!latent) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "%s", iris_get_error());
            return;
        }

        /* Encode latent as base64 */
        int latent_ch = 128; /* IRIS_LATENT_CHANNELS */
        size_t latent_bytes = (size_t)latent_ch * out_h * out_w * sizeof(float);
        size_t b64_len;
        char *b64 = b64_encode((const unsigned char *)latent, latent_bytes, &b64_len);
        free(latent);

        if (!b64) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to encode latent");
            return;
        }

        size_t json_len = b64_len + 256;
        job->result_json = malloc(json_len);
        snprintf(job->result_json, json_len,
                 "{\"latent\":\"%s\",\"latent_h\":%d,\"latent_w\":%d}",
                 b64, out_h, out_w);
        free(b64);
        job->state = JOB_COMPLETE;
        break;
    }

    case JOB_VAE_DECODE: {
        iris_image *img = iris_decode_latent(ctx, job->latent,
                                             job->latent_h, job->latent_w);
        if (!img) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "%s", iris_get_error());
            return;
        }
        job->result_width = img->width;
        job->result_height = img->height;
        job->result_png = iris_image_to_png_mem(img, &job->result_png_len, seed);
        iris_image_free(img);
        if (!job->result_png) {
            job->state = JOB_FAILED;
            snprintf(job->error_msg, sizeof(job->error_msg), "Failed to encode PNG");
            return;
        }
        job->state = JOB_COMPLETE;
        break;
    }
    }
}

static void *worker_thread(void *arg) {
    worker_ctx_t *wctx = (worker_ctx_t *)arg;
    job_queue_t *q = wctx->queue;

    /* Disable progress callbacks (no terminal in server mode) */
    iris_step_callback = NULL;
    iris_substep_callback = NULL;
    iris_phase_callback = NULL;

    while (1) {
        pthread_mutex_lock(&q->mutex);

        /* Wait for a job or shutdown */
        int slot;
        while ((slot = job_dequeue(q)) < 0 && !q->shutdown) {
            pthread_cond_wait(&q->cond, &q->mutex);
        }

        if (q->shutdown && slot < 0) {
            pthread_mutex_unlock(&q->mutex);
            break;
        }

        http_job_t *job = &q->jobs[slot];
        pthread_mutex_unlock(&q->mutex);

        /* Process the job (this can take seconds to minutes) */
        fprintf(stderr, "[worker] Processing job %d (type=%d)\n", job->id, job->type);
        worker_process_job(wctx->ctx, job);

        pthread_mutex_lock(&q->mutex);
        if (job->state != JOB_FAILED) job->state = JOB_COMPLETE;
        job->completed_at = time(NULL);
        pthread_mutex_unlock(&q->mutex);

        fprintf(stderr, "[worker] Job %d %s\n", job->id,
                job->state == JOB_COMPLETE ? "complete" : "failed");
    }

    return NULL;
}

/* ========================================================================
 * Route Handlers
 * ======================================================================== */

static void handle_health(int fd) {
    http_respond_json(fd, 200, "OK", "{\"status\":\"ok\"}");
}

static void handle_info(int fd, iris_ctx *ctx) {
    char json[1024];
    snprintf(json, sizeof(json),
        "{\"model\":\"%s\",\"is_distilled\":%s,\"is_zimage\":%s,"
        "\"text_dim\":%d,\"is_non_commercial\":%s}",
        iris_model_info(ctx),
        iris_is_distilled(ctx) ? "true" : "false",
        iris_is_zimage(ctx) ? "true" : "false",
        iris_text_dim(ctx),
        iris_is_non_commercial(ctx) ? "true" : "false");
    http_respond_json(fd, 200, "OK", json);
}

static void handle_generate(int fd, job_queue_t *q, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        http_respond_error(fd, 400, "Request body required");
        return;
    }

    json_obj_t *obj = json_parse(req->body, req->body_len);
    if (!obj) {
        http_respond_error(fd, 400, "Invalid JSON");
        return;
    }

    const char *prompt = json_get_string(obj, "prompt");
    if (!prompt) {
        json_free(obj);
        http_respond_error(fd, 400, "prompt is required");
        return;
    }

    iris_params params = params_from_json(obj);
    const char *err = validate_params(&params);
    if (err) {
        json_free(obj);
        http_respond_error(fd, 400, err);
        return;
    }

    http_job_t job = {0};
    job.type = JOB_GENERATE;
    job.prompt = strdup(prompt);
    job.params = params;
    job.seed = params.seed;

    json_free(obj);

    int id = job_enqueue(q, &job);
    if (id < 0) {
        free(job.prompt);
        http_respond_error(fd, 503, "Job queue full");
        return;
    }

    char json[128];
    snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"pending\"}", id);
    http_respond_json(fd, 202, "Accepted", json);
}

static void handle_img2img(int fd, job_queue_t *q, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        http_respond_error(fd, 400, "Request body required");
        return;
    }

    json_obj_t *obj = json_parse(req->body, req->body_len);
    if (!obj) {
        http_respond_error(fd, 400, "Invalid JSON");
        return;
    }

    const char *prompt = json_get_string(obj, "prompt");
    const char *image_b64 = json_get_string(obj, "image");
    if (!prompt || !image_b64) {
        json_free(obj);
        http_respond_error(fd, 400, "prompt and image are required");
        return;
    }

    iris_params params = params_from_json(obj);
    const char *err = validate_params(&params);
    if (err) {
        json_free(obj);
        http_respond_error(fd, 400, err);
        return;
    }

    http_job_t job = {0};
    job.type = JOB_IMG2IMG;
    job.prompt = strdup(prompt);
    job.params = params;
    job.seed = params.seed;
    /* Store the base64 string directly; worker will decode it */
    job.input_image_data = (uint8_t *)strdup(image_b64);
    job.input_image_len = strlen(image_b64);

    json_free(obj);

    int id = job_enqueue(q, &job);
    if (id < 0) {
        free(job.prompt);
        free(job.input_image_data);
        http_respond_error(fd, 503, "Job queue full");
        return;
    }

    char json[128];
    snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"pending\"}", id);
    http_respond_json(fd, 202, "Accepted", json);
}

static void handle_multiref(int fd, job_queue_t *q, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        http_respond_error(fd, 400, "Request body required");
        return;
    }

    json_obj_t *obj = json_parse(req->body, req->body_len);
    if (!obj) {
        http_respond_error(fd, 400, "Invalid JSON");
        return;
    }

    const char *prompt = json_get_string(obj, "prompt");
    char **images = NULL;
    int num_images = 0;
    json_get_string_array(obj, "images", &images, &num_images);

    if (!prompt || num_images == 0) {
        json_free(obj);
        http_respond_error(fd, 400, "prompt and images array are required");
        return;
    }
    if (num_images > MAX_REFS) {
        json_free(obj);
        char msg[64];
        snprintf(msg, sizeof(msg), "Maximum %d reference images", MAX_REFS);
        http_respond_error(fd, 400, msg);
        return;
    }

    iris_params params = params_from_json(obj);
    const char *err = validate_params(&params);
    if (err) {
        json_free(obj);
        http_respond_error(fd, 400, err);
        return;
    }

    http_job_t job = {0};
    job.type = JOB_MULTIREF;
    job.prompt = strdup(prompt);
    job.params = params;
    job.seed = params.seed;
    job.num_refs = num_images;
    job.ref_images_data = calloc(num_images, sizeof(uint8_t *));
    job.ref_images_lens = calloc(num_images, sizeof(size_t));
    for (int i = 0; i < num_images; i++) {
        job.ref_images_data[i] = (uint8_t *)strdup(images[i]);
        job.ref_images_lens[i] = strlen(images[i]);
    }

    json_free(obj);

    int id = job_enqueue(q, &job);
    if (id < 0) {
        free(job.prompt);
        for (int i = 0; i < num_images; i++) free(job.ref_images_data[i]);
        free(job.ref_images_data);
        free(job.ref_images_lens);
        http_respond_error(fd, 503, "Job queue full");
        return;
    }

    char json[128];
    snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"pending\"}", id);
    http_respond_json(fd, 202, "Accepted", json);
}

static void handle_encode_text(int fd, job_queue_t *q, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        http_respond_error(fd, 400, "Request body required");
        return;
    }

    json_obj_t *obj = json_parse(req->body, req->body_len);
    if (!obj) {
        http_respond_error(fd, 400, "Invalid JSON");
        return;
    }

    const char *prompt = json_get_string(obj, "prompt");
    if (!prompt) {
        json_free(obj);
        http_respond_error(fd, 400, "prompt is required");
        return;
    }

    http_job_t job = {0};
    job.type = JOB_ENCODE_TEXT;
    job.prompt = strdup(prompt);

    json_free(obj);

    int id = job_enqueue(q, &job);
    if (id < 0) {
        free(job.prompt);
        http_respond_error(fd, 503, "Job queue full");
        return;
    }

    char json[128];
    snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"pending\"}", id);
    http_respond_json(fd, 202, "Accepted", json);
}

static void handle_generate_with_embeddings(int fd, job_queue_t *q, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        http_respond_error(fd, 400, "Request body required");
        return;
    }

    json_obj_t *obj = json_parse(req->body, req->body_len);
    if (!obj) {
        http_respond_error(fd, 400, "Invalid JSON");
        return;
    }

    const char *emb_b64 = json_get_string(obj, "embeddings");
    int seq_len = json_get_int(obj, "seq_len", 0);
    if (!emb_b64 || seq_len <= 0) {
        json_free(obj);
        http_respond_error(fd, 400, "embeddings and seq_len are required");
        return;
    }

    iris_params params = params_from_json(obj);
    const char *err = validate_params(&params);
    if (err) {
        json_free(obj);
        http_respond_error(fd, 400, err);
        return;
    }

    /* Decode embeddings */
    size_t emb_raw_len;
    uint8_t *emb_raw = b64_decode(emb_b64, strlen(emb_b64), &emb_raw_len);
    json_free(obj);

    if (!emb_raw) {
        http_respond_error(fd, 400, "Failed to decode embeddings");
        return;
    }

    http_job_t job = {0};
    job.type = JOB_GENERATE_WITH_EMBEDDINGS;
    job.params = params;
    job.seed = params.seed;
    job.embeddings = (float *)emb_raw;
    job.emb_seq_len = seq_len;

    int id = job_enqueue(q, &job);
    if (id < 0) {
        free(emb_raw);
        http_respond_error(fd, 503, "Job queue full");
        return;
    }

    char json[128];
    snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"pending\"}", id);
    http_respond_json(fd, 202, "Accepted", json);
}

static void handle_vae_encode(int fd, job_queue_t *q, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        http_respond_error(fd, 400, "Request body required");
        return;
    }

    json_obj_t *obj = json_parse(req->body, req->body_len);
    if (!obj) {
        http_respond_error(fd, 400, "Invalid JSON");
        return;
    }

    const char *image_b64 = json_get_string(obj, "image");
    if (!image_b64) {
        json_free(obj);
        http_respond_error(fd, 400, "image is required");
        return;
    }

    http_job_t job = {0};
    job.type = JOB_VAE_ENCODE;
    job.input_image_data = (uint8_t *)strdup(image_b64);
    job.input_image_len = strlen(image_b64);

    json_free(obj);

    int id = job_enqueue(q, &job);
    if (id < 0) {
        free(job.input_image_data);
        http_respond_error(fd, 503, "Job queue full");
        return;
    }

    char json[128];
    snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"pending\"}", id);
    http_respond_json(fd, 202, "Accepted", json);
}

static void handle_vae_decode(int fd, job_queue_t *q, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        http_respond_error(fd, 400, "Request body required");
        return;
    }

    json_obj_t *obj = json_parse(req->body, req->body_len);
    if (!obj) {
        http_respond_error(fd, 400, "Invalid JSON");
        return;
    }

    const char *latent_b64 = json_get_string(obj, "latent");
    int latent_h = json_get_int(obj, "latent_h", 0);
    int latent_w = json_get_int(obj, "latent_w", 0);
    if (!latent_b64 || latent_h <= 0 || latent_w <= 0) {
        json_free(obj);
        http_respond_error(fd, 400, "latent, latent_h, and latent_w are required");
        return;
    }

    /* Decode latent */
    size_t latent_raw_len;
    uint8_t *latent_raw = b64_decode(latent_b64, strlen(latent_b64), &latent_raw_len);
    json_free(obj);

    if (!latent_raw) {
        http_respond_error(fd, 400, "Failed to decode latent");
        return;
    }

    http_job_t job = {0};
    job.type = JOB_VAE_DECODE;
    job.latent = (float *)latent_raw;
    job.latent_h = latent_h;
    job.latent_w = latent_w;
    job.seed = -1;

    int id = job_enqueue(q, &job);
    if (id < 0) {
        free(latent_raw);
        http_respond_error(fd, 503, "Job queue full");
        return;
    }

    char json[128];
    snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"pending\"}", id);
    http_respond_json(fd, 202, "Accepted", json);
}

static void handle_job_status(int fd, job_queue_t *q, int job_id, const char *query) {
    pthread_mutex_lock(&q->mutex);
    int slot = job_find(q, job_id);
    if (slot < 0) {
        pthread_mutex_unlock(&q->mutex);
        http_respond_error(fd, 404, "Job not found");
        return;
    }

    http_job_t *job = &q->jobs[slot];
    int state = job->state;

    if (state == JOB_PENDING || state == JOB_RUNNING) {
        pthread_mutex_unlock(&q->mutex);
        char json[128];
        snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"%s\"}",
                 job_id, state == JOB_PENDING ? "pending" : "running");
        http_respond_json(fd, 200, "OK", json);
        return;
    }

    if (state == JOB_FAILED) {
        char json[512];
        snprintf(json, sizeof(json), "{\"job_id\":%d,\"status\":\"failed\",\"error\":\"%s\"}",
                 job_id, job->error_msg);
        pthread_mutex_unlock(&q->mutex);
        http_respond_json(fd, 200, "OK", json);
        return;
    }

    /* JOB_COMPLETE */
    if (job->result_json) {
        /* Non-image result (encode-text, vae/encode) */
        /* Wrap the inner JSON with job metadata */
        size_t inner_len = strlen(job->result_json);
        size_t json_len = inner_len + 256;
        char *json = malloc(json_len);
        /* The result_json is already a complete object; inject job_id and status */
        snprintf(json, json_len,
                 "{\"job_id\":%d,\"status\":\"complete\",%s",
                 job_id, job->result_json + 1); /* skip opening { */
        pthread_mutex_unlock(&q->mutex);
        http_respond_json(fd, 200, "OK", json);
        free(json);
        return;
    }

    if (job->result_png) {
        int want_json = query_has(query, "format", "json");

        if (want_json) {
            size_t b64_len;
            char *b64 = b64_encode(job->result_png, job->result_png_len, &b64_len);
            pthread_mutex_unlock(&q->mutex);

            if (!b64) {
                http_respond_error(fd, 500, "Failed to encode result");
                return;
            }

            size_t json_len = b64_len + 256;
            char *json = malloc(json_len);
            snprintf(json, json_len,
                     "{\"job_id\":%d,\"status\":\"complete\",\"image\":\"%s\","
                     "\"seed\":%lld,\"width\":%d,\"height\":%d}",
                     job_id, b64, (long long)job->result_seed,
                     job->result_width, job->result_height);
            http_respond_json(fd, 200, "OK", json);
            free(json);
            free(b64);
        } else {
            /* Return raw PNG */
            /* Copy data before releasing mutex in case job expires */
            size_t png_len = job->result_png_len;
            uint8_t *png_copy = malloc(png_len);
            memcpy(png_copy, job->result_png, png_len);
            pthread_mutex_unlock(&q->mutex);
            http_respond_png(fd, png_copy, png_len);
            free(png_copy);
        }
        return;
    }

    pthread_mutex_unlock(&q->mutex);
    http_respond_error(fd, 500, "Job complete but no result");
}

static void handle_job_delete(int fd, job_queue_t *q, int job_id) {
    pthread_mutex_lock(&q->mutex);
    int slot = job_find(q, job_id);
    if (slot < 0) {
        pthread_mutex_unlock(&q->mutex);
        http_respond_error(fd, 404, "Job not found");
        return;
    }

    http_job_t *job = &q->jobs[slot];

    /* Only allow deleting completed/failed jobs */
    if (job->state == JOB_PENDING || job->state == JOB_RUNNING) {
        pthread_mutex_unlock(&q->mutex);
        http_respond_error(fd, 400, "Cannot delete active job");
        return;
    }

    /* Free resources */
    free(job->prompt); job->prompt = NULL;
    free(job->input_image_data); job->input_image_data = NULL;
    free(job->result_png); job->result_png = NULL;
    free(job->result_json); job->result_json = NULL;
    free(job->embeddings); job->embeddings = NULL;
    free(job->latent); job->latent = NULL;
    if (job->ref_images_data) {
        for (int i = 0; i < job->num_refs; i++) free(job->ref_images_data[i]);
        free(job->ref_images_data); job->ref_images_data = NULL;
        free(job->ref_images_lens); job->ref_images_lens = NULL;
    }
    job->active = 0;

    pthread_mutex_unlock(&q->mutex);
    http_respond_json(fd, 200, "OK", "{\"deleted\":true}");
}

/* Parse job ID from path like "/jobs/123" */
static int parse_job_id(const char *path) {
    if (strncmp(path, "/jobs/", 6) != 0) return -1;
    const char *id_str = path + 6;
    if (*id_str < '0' || *id_str > '9') return -1;
    return atoi(id_str);
}

/* ========================================================================
 * Server Main Loop
 * ======================================================================== */

static volatile sig_atomic_t server_shutdown = 0;

static void signal_handler(int sig) {
    (void)sig;
    server_shutdown = 1;
}

int iris_http_serve(iris_ctx *ctx, int port) {
    /* Set up signal handlers */
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    /* Create listen socket */
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return -1;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(listen_fd);
        return -1;
    }

    if (listen(listen_fd, 32) < 0) {
        perror("listen");
        close(listen_fd);
        return -1;
    }

    fprintf(stderr, "Iris HTTP server listening on port %d\n", port);
    fprintf(stderr, "Endpoints:\n");
    fprintf(stderr, "  GET  /health\n");
    fprintf(stderr, "  GET  /info\n");
    fprintf(stderr, "  POST /generate\n");
    fprintf(stderr, "  POST /img2img\n");
    fprintf(stderr, "  POST /multiref\n");
    fprintf(stderr, "  POST /encode-text\n");
    fprintf(stderr, "  POST /generate-with-embeddings\n");
    fprintf(stderr, "  POST /vae/encode\n");
    fprintf(stderr, "  POST /vae/decode\n");
    fprintf(stderr, "  GET  /jobs/{id}\n");
    fprintf(stderr, "  DELETE /jobs/{id}\n\n");

    /* Initialize job queue */
    job_queue_t queue;
    job_queue_init(&queue);

    /* Start worker thread */
    worker_ctx_t wctx = { .ctx = ctx, .queue = &queue };
    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_thread, &wctx) != 0) {
        perror("pthread_create");
        close(listen_fd);
        return -1;
    }

    /* Accept loop */
    while (!server_shutdown) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(listen_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        /* Set read timeout */
        struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        /* Read and parse request */
        http_request_t req;
        if (http_read_request(client_fd, &req) < 0) {
            close(client_fd);
            continue;
        }

        fprintf(stderr, "[http] %s %s\n", req.method, req.path);

        /* CORS preflight */
        if (strcmp(req.method, "OPTIONS") == 0) {
            http_respond_raw(client_fd, 204, "No Content", "text/plain", NULL, 0);
            goto next;
        }

        /* Route dispatch */
        if (strcmp(req.path, "/health") == 0 && strcmp(req.method, "GET") == 0) {
            handle_health(client_fd);
        }
        else if (strcmp(req.path, "/info") == 0 && strcmp(req.method, "GET") == 0) {
            handle_info(client_fd, ctx);
        }
        else if (strcmp(req.path, "/generate") == 0 && strcmp(req.method, "POST") == 0) {
            handle_generate(client_fd, &queue, &req);
        }
        else if (strcmp(req.path, "/img2img") == 0 && strcmp(req.method, "POST") == 0) {
            handle_img2img(client_fd, &queue, &req);
        }
        else if (strcmp(req.path, "/multiref") == 0 && strcmp(req.method, "POST") == 0) {
            handle_multiref(client_fd, &queue, &req);
        }
        else if (strcmp(req.path, "/encode-text") == 0 && strcmp(req.method, "POST") == 0) {
            handle_encode_text(client_fd, &queue, &req);
        }
        else if (strcmp(req.path, "/generate-with-embeddings") == 0
                 && strcmp(req.method, "POST") == 0) {
            handle_generate_with_embeddings(client_fd, &queue, &req);
        }
        else if (strcmp(req.path, "/vae/encode") == 0 && strcmp(req.method, "POST") == 0) {
            handle_vae_encode(client_fd, &queue, &req);
        }
        else if (strcmp(req.path, "/vae/decode") == 0 && strcmp(req.method, "POST") == 0) {
            handle_vae_decode(client_fd, &queue, &req);
        }
        else if (strncmp(req.path, "/jobs/", 6) == 0) {
            int job_id = parse_job_id(req.path);
            if (job_id < 0) {
                http_respond_error(client_fd, 400, "Invalid job ID");
            } else if (strcmp(req.method, "GET") == 0) {
                handle_job_status(client_fd, &queue, job_id, req.query);
            } else if (strcmp(req.method, "DELETE") == 0) {
                handle_job_delete(client_fd, &queue, job_id);
            } else {
                http_respond_error(client_fd, 405, "Method not allowed");
            }
        }
        else {
            http_respond_error(client_fd, 404, "Not found");
        }

    next:
        http_request_free(&req);
        close(client_fd);
    }

    fprintf(stderr, "\nShutting down...\n");

    /* Signal worker to stop */
    pthread_mutex_lock(&queue.mutex);
    queue.shutdown = 1;
    pthread_cond_signal(&queue.cond);
    pthread_mutex_unlock(&queue.mutex);
    pthread_join(worker, NULL);

    close(listen_fd);
    job_queue_destroy(&queue);

    return 0;
}
