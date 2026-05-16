/*
 * Unit tests for nvrtc_stubs.c
 * Compile with: gcc -o test_nvrtc_stubs test_nvrtc_stubs.c nvrtc_stubs.c
 * On A100: gcc -o test_nvrtc_stubs test_nvrtc_stubs.c nvrtc_stubs.c \
 *          -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvrtc -lcuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "nvrtc.h"

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
  tests_run++; \
  if (cond) { \
    tests_passed++; \
    printf("[OK]  %s\n", msg); \
  } else { \
    printf("[FAIL] %s\n", msg); \
  } \
} while(0)

static const char* trivial_source =
  "extern \"C\" __global__ void trivial(float* out, int n) {\n"
  "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
  "  if (idx < n) out[idx] = 1.0f;\n"
  "}\n";

static const char* invalid_source =
  "this is not valid cuda c++ {\n"
  "  __global__ void broken(\n";

/* ------------------------------------------------------------------ */
/* create / destroy                                                    */
/* ------------------------------------------------------------------ */

void test_create_returns_nonnull() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  CHECK(prog != NULL, "create returns non-null");
  tesserae_nvrtc_destroy(prog);
}

void test_is_valid_after_create() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  CHECK(tesserae_nvrtc_is_valid(prog) == 1, "valid after create");
  tesserae_nvrtc_destroy(prog);
}

void test_is_invalid_after_destroy() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  tesserae_nvrtc_destroy(prog);
  CHECK(tesserae_nvrtc_is_valid(prog) == 0, "invalid after destroy");
}

void test_destroy_idempotent() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  tesserae_nvrtc_destroy(prog);
  tesserae_nvrtc_destroy(prog); /* should not crash */
  CHECK(1, "destroy idempotent");
}

/* ------------------------------------------------------------------ */
/* compile                                                             */
/* ------------------------------------------------------------------ */

void test_compile_trivial() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  const char* opts[] = {"--gpu-architecture=sm_80", NULL};
  int rc = tesserae_nvrtc_compile(prog, opts, 1);
  CHECK(rc == 0, "compile trivial returns 0");
  tesserae_nvrtc_destroy(prog);
}

void test_compile_invalid_source() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(invalid_source, "broken.cu");
  const char* opts[] = {"--gpu-architecture=sm_80", NULL};
  int rc = tesserae_nvrtc_compile(prog, opts, 1);
  CHECK(rc != 0, "compile invalid returns nonzero");
  tesserae_nvrtc_destroy(prog);
}

/* ------------------------------------------------------------------ */
/* get_ptx                                                             */
/* ------------------------------------------------------------------ */

void test_get_ptx_nonempty() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  const char* opts[] = {"--gpu-architecture=sm_80", NULL};
  int rc = tesserae_nvrtc_compile(prog, opts, 1);
  if (rc != 0) {
    printf("[SKIP] test_get_ptx_nonempty (compile failed)\n");
    tesserae_nvrtc_destroy(prog);
    return;
  }
  char* ptx = tesserae_nvrtc_get_ptx(prog);
  CHECK(ptx != NULL, "get_ptx non-null");
  CHECK(strlen(ptx) > 0, "get_ptx non-empty");
  CHECK(strstr(ptx, ".visible") != NULL, "ptx has .visible");
  CHECK(strstr(ptx, ".entry")   != NULL, "ptx has .entry");
  free(ptx);
  tesserae_nvrtc_destroy(prog);
}

void test_get_ptx_before_compile() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  char* ptx = tesserae_nvrtc_get_ptx(prog);
  CHECK(ptx == NULL, "get_ptx before compile returns null");
  tesserae_nvrtc_destroy(prog);
}

/* ------------------------------------------------------------------ */
/* get_log                                                             */
/* ------------------------------------------------------------------ */

void test_get_log_on_error() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(invalid_source, "broken.cu");
  const char* opts[] = {"--gpu-architecture=sm_80", NULL};
  tesserae_nvrtc_compile(prog, opts, 1);
  char* log = tesserae_nvrtc_get_log(prog);
  CHECK(log != NULL,      "error log non-null");
  CHECK(strlen(log) > 0,  "error log non-empty");
  free(log);
  tesserae_nvrtc_destroy(prog);
}

void test_get_log_on_success() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  const char* opts[] = {"--gpu-architecture=sm_80", NULL};
  int rc = tesserae_nvrtc_compile(prog, opts, 1);
  if (rc != 0) {
    printf("[SKIP] test_get_log_on_success\n");
    tesserae_nvrtc_destroy(prog);
    return;
  }
  char* log = tesserae_nvrtc_get_log(prog);
  /* log may be empty string on success — just check non-null */
  CHECK(log != NULL, "success log non-null");
  free(log);
  tesserae_nvrtc_destroy(prog);
}

/* ------------------------------------------------------------------ */
/* options                                                             */
/* ------------------------------------------------------------------ */

void test_compile_fast_math() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  const char* opts[] = {
    "--gpu-architecture=sm_80",
    "--use_fast_math",
    "--generate-line-info",
    NULL
  };
  int rc = tesserae_nvrtc_compile(prog, opts, 3);
  CHECK(rc == 0, "compile with fast math");
  tesserae_nvrtc_destroy(prog);
}

void test_compile_no_options() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  int rc = tesserae_nvrtc_compile(prog, NULL, 0);
  /* may succeed or fail depending on default arch — just no crash */
  CHECK(rc == 0 || rc != 0, "compile no options does not crash");
  tesserae_nvrtc_destroy(prog);
}

/* ------------------------------------------------------------------ */
/* source / name accessors                                             */
/* ------------------------------------------------------------------ */

void test_source_preserved() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  const char* src = tesserae_nvrtc_get_source(prog);
  CHECK(strcmp(src, trivial_source) == 0, "source preserved");
  tesserae_nvrtc_destroy(prog);
}

void test_name_preserved() {
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(trivial_source, "trivial.cu");
  const char* name = tesserae_nvrtc_get_name(prog);
  CHECK(strcmp(name, "trivial.cu") == 0, "name preserved");
  tesserae_nvrtc_destroy(prog);
}

/* ------------------------------------------------------------------ */
/* runner                                                              */
/* ------------------------------------------------------------------ */

int main(void) {
  printf("=== nvrtc_stubs tests ===\n\n");

  printf("-- create/destroy --\n");
  test_create_returns_nonnull();
  test_is_valid_after_create();
  test_is_invalid_after_destroy();
  test_destroy_idempotent();

  printf("\n-- compile --\n");
  test_compile_trivial();
  test_compile_invalid_source();

  printf("\n-- ptx --\n");
  test_get_ptx_nonempty();
  test_get_ptx_before_compile();

  printf("\n-- log --\n");
  test_get_log_on_error();
  test_get_log_on_success();

  printf("\n-- options --\n");
  test_compile_fast_math();
  test_compile_no_options();

  printf("\n-- accessors --\n");
  test_source_preserved();
  test_name_preserved();

  printf("\n=== %d/%d passed ===\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
