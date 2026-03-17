# Iris - C Image Generation Engine
# Supported models: FLUX.2 Klein (4B/9B), Z-Image-Turbo (6B)
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Source files
SRCS = iris.c iris_kernels.c iris_tokenizer.c iris_vae.c iris_transformer_flux.c iris_transformer_zimage.c iris_sample.c iris_image.c jpeg.c iris_safetensors.c iris_qwen3.c iris_qwen3_tokenizer.c terminals.c
OBJS = $(SRCS:.c=.o)
CLI_SRCS = iris_cli.c linenoise.c embcache.c
CLI_OBJS = $(CLI_SRCS:.c=.o)
HTTP_SRCS = iris_http.c
HTTP_OBJS = $(HTTP_SRCS:.c=.o)
MAIN = main.c
TARGET = iris
SERVER_TARGET = iris-server
LIB = libiris.a

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug lib install info test pngtest help generic blas mps serve-generic serve-blas serve-mps
.NOTPARALLEL: mps serve-mps

# Default: show available targets
all: help

help:
	@echo "Iris - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make generic  - Pure C, no dependencies (slow)"
	@echo "  make blas     - With BLAS acceleration (~30x faster)"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  make mps      - Apple Silicon with Metal GPU (fastest)"
endif
endif
	@echo ""
	@echo "HTTP server:"
	@echo "  make serve-generic - HTTP server, pure C backend"
	@echo "  make serve-blas    - HTTP server, BLAS backend"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  make serve-mps     - HTTP server, Metal GPU backend"
endif
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Run inference test"
	@echo "  make pngtest  - Compare PNG load on compressed image"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""
	@echo "Example: make mps && ./iris -d flux-klein-4b -p \"a cat\" -o cat.png"
	@echo "         make serve-blas && ./iris-server -d flux-klein-4b --port 8080"

# =============================================================================
# Backend: generic (pure C, no BLAS)
# =============================================================================
generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
generic: clean $(TARGET)
	@echo ""
	@echo "Built with GENERIC backend (pure C, no BLAS)"
	@echo "This will be slow but has zero dependencies."

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend (~30x faster than generic)"

# =============================================================================
# Backend: mps (Apple Silicon Metal GPU)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
MPS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_METAL -DACCELERATE_NEW_LAPACK
MPS_OBJCFLAGS = $(MPS_CFLAGS) -fobjc-arc
MPS_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation

mps: clean mps-build
	@echo ""
	@echo "Built with MPS backend (Metal GPU acceleration)"

mps-build: $(SRCS:.c=.mps.o) $(CLI_SRCS:.c=.mps.o) iris_metal.o main.mps.o
	$(CC) $(MPS_CFLAGS) -o $(TARGET) $^ $(MPS_LDFLAGS)

%.mps.o: %.c iris.h iris_kernels.h
	$(CC) $(MPS_CFLAGS) -c -o $@ $<

# Embed Metal shader source as C array (runtime compilation, no Metal toolchain needed)
iris_shaders_source.h: iris_shaders.metal
	xxd -i $< > $@

iris_metal.o: iris_metal.m iris_metal.h iris_shaders_source.h
	$(CC) $(MPS_OBJCFLAGS) -c -o $@ $<

else
mps:
	@echo "Error: MPS backend requires Apple Silicon (arm64)"
	@exit 1
endif
else
mps:
	@echo "Error: MPS backend requires macOS"
	@exit 1
endif

# =============================================================================
# HTTP Server: generic
# =============================================================================
serve-generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
serve-generic: LDFLAGS += -lpthread
serve-generic: clean $(SERVER_TARGET)
	@echo ""
	@echo "Built iris-server with GENERIC backend"

# =============================================================================
# HTTP Server: blas
# =============================================================================
ifeq ($(UNAME_S),Darwin)
serve-blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
serve-blas: LDFLAGS += -framework Accelerate -lpthread
else
serve-blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
serve-blas: LDFLAGS += -lopenblas -lpthread
endif
serve-blas: clean $(SERVER_TARGET)
	@echo ""
	@echo "Built iris-server with BLAS backend"

# =============================================================================
# HTTP Server: mps
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
serve-mps: clean serve-mps-build
	@echo ""
	@echo "Built iris-server with MPS backend"

serve-mps-build: $(SRCS:.c=.mps.o) $(HTTP_SRCS:.c=.mps.o) iris_metal.o main_http.mps.o
	$(CC) $(MPS_CFLAGS) -o $(SERVER_TARGET) $^ $(MPS_LDFLAGS) -lpthread

else
serve-mps:
	@echo "Error: MPS backend requires Apple Silicon (arm64)"
	@exit 1
endif
else
serve-mps:
	@echo "Error: MPS backend requires macOS"
	@exit 1
endif

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) $(CLI_OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SERVER_TARGET): $(OBJS) $(HTTP_OBJS) main_http.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c iris.h iris_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# =============================================================================
# Test and utilities
# =============================================================================
test:
	@python3 run_test.py --flux-binary ./$(TARGET)

test-quick:
	@python3 run_test.py --flux-binary ./$(TARGET) --quick

pngtest:
	@echo "Running PNG compression compare test..."
	@$(CC) $(CFLAGS_BASE) -I. png_compare.c iris_image.c -lm -o /tmp/iris_png_compare
	@/tmp/iris_png_compare images/woman_with_sunglasses.png images/woman_with_sunglasses_compressed2.png
	@/tmp/iris_png_compare images/cat_uncompressed.png images/cat_compressed.png
	@rm -f /tmp/iris_png_compare
	@echo "PNG TEST PASSED"

install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 iris.h /usr/local/include/
	install -m 644 iris_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) $(CLI_OBJS) $(HTTP_OBJS) *.mps.o iris_metal.o main.o main_http.o $(TARGET) $(SERVER_TARGET) $(LIB)
	rm -f iris_shaders_source.h

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo ""
	@echo "Available backends for this platform:"
	@echo "  generic - Pure C (always available)"
ifeq ($(UNAME_S),Darwin)
	@echo "  blas    - Apple Accelerate"
ifeq ($(UNAME_M),arm64)
	@echo "  mps     - Metal GPU (recommended)"
endif
else
	@echo "  blas    - OpenBLAS (requires libopenblas-dev)"
endif

# =============================================================================
# Dependencies
# =============================================================================
iris.o: iris.c iris.h iris_kernels.h iris_safetensors.h iris_qwen3.h
iris_kernels.o: iris_kernels.c iris_kernels.h
iris_tokenizer.o: iris_tokenizer.c iris.h
iris_vae.o: iris_vae.c iris.h iris_kernels.h
iris_transformer_flux.o: iris_transformer_flux.c iris.h iris_kernels.h
iris_transformer_zimage.o: iris_transformer_zimage.c iris.h iris_kernels.h iris_safetensors.h
iris_sample.o: iris_sample.c iris.h iris_kernels.h
iris_image.o: iris_image.c iris.h
iris_safetensors.o: iris_safetensors.c iris_safetensors.h
iris_qwen3.o: iris_qwen3.c iris_qwen3.h iris_safetensors.h
iris_qwen3_tokenizer.o: iris_qwen3_tokenizer.c iris_qwen3.h
terminals.o: terminals.c terminals.h iris.h
iris_cli.o: iris_cli.c iris_cli.h iris.h iris_qwen3.h embcache.h linenoise.h terminals.h
linenoise.o: linenoise.c linenoise.h
embcache.o: embcache.c embcache.h
main.o: main.c iris.h iris_kernels.h iris_cli.h terminals.h
iris_http.o: iris_http.c iris_http.h iris.h
main_http.o: main_http.c iris.h iris_http.h
