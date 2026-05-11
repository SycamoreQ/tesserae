#include <stdlib.h>
#include <string.h>
#include "nvrtc_stub.h"
#include <nvrtc.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/custom.h>


static void nvrtc_program_finalize(value v) {
  tesserae_nvrtc_program* prog =
    *((tesserae_nvrtc_program**) Data_custom_val(v));
  if (prog) tesserae_nvrtc_destroy(prog);
}

static struct custom_operations nvrtc_program_ops = {
  "tesserae_nvrtc_program",
  nvrtc_program_finalize,
  custom_compare_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default,
  custom_compare_ext_default,
  custom_fixed_length_default
};

tesserae_nvrtc_program* tesserae_nvrtc_create(
    const char* source, const char* name) {
  tesserae_nvrtc_program* prog =
    malloc(sizeof(tesserae_nvrtc_program));
  if (!prog) return NULL;
  prog->source = strdup(source);
  prog->name = strdup(name);
  prog->ptx = NULL;
  prog->log = NULL;
  prog->compiled = 0;
  prog->valid = 1;
  prog->handle = NULL;
  return prog;
}

void tesserae_nvrtc_destroy(tesserae_nvrtc_program* prog) {
  if (!prog || !prog->valid) return;
  prog->valid = 0;
  if (prog->handle)
    nvrtcDestroyProgram((nvrtcProgram*) &prog->handle);
  free(prog->source);
  free(prog->name);
  free(prog->ptx);
  free(prog->log);
}

int tesserae_nvrtc_is_valid(tesserae_nvrtc_program* prog) {
  return prog && prog->valid;
}

const char* tesserae_nvrtc_get_source(tesserae_nvrtc_program* prog) {
  return prog && prog->valid ? prog->source : NULL;
}

const char* tesserae_nvrtc_get_name(tesserae_nvrtc_program* prog) {
  return prog && prog->valid ? prog->name : NULL;
}

int tesserae_nvrtc_compile(
    tesserae_nvrtc_program* prog,
    const char** opts, int n_opts) {
  if (!prog || !prog->valid) return -1;

  nvrtcResult rc = nvrtcCreateProgram(
    (nvrtcProgram*) &prog->handle,
    prog->source, prog->name,
    0, NULL, NULL);
  if (rc != NVRTC_SUCCESS) return (int) rc;

  rc = nvrtcCompileProgram((nvrtcProgram) prog->handle, n_opts, opts);

  size_t log_size = 0;
  nvrtcGetProgramLogSize((nvrtcProgram) prog->handle, &log_size);
  prog->log = malloc(log_size);
  if (prog->log)
    nvrtcGetProgramLog((nvrtcProgram) prog->handle, prog->log);

  prog->compiled = 1;
  if (rc != NVRTC_SUCCESS) return (int) rc;

  size_t ptx_size = 0;
  nvrtcGetPTXSize((nvrtcProgram) prog->handle, &ptx_size);
  prog->ptx = malloc(ptx_size);
  if (prog->ptx)
    nvrtcGetPTX((nvrtcProgram) prog->handle, prog->ptx);

  return 0;
}

char* tesserae_nvrtc_get_ptx(tesserae_nvrtc_program* prog) {
  if (!prog || !prog->valid || !prog->ptx) return NULL;
  return strdup(prog->ptx);
}

char* tesserae_nvrtc_get_log(tesserae_nvrtc_program* prog) {
  if (!prog || !prog->valid || !prog->log) return NULL;
  return strdup(prog->log);
}


CAMLprim value caml_nvrtc_create(value source, value name) {
  CAMLparam2(source, name);
  CAMLlocal1(block);
  tesserae_nvrtc_program* prog =
    tesserae_nvrtc_create(String_val(source), String_val(name));
  if (!prog) caml_failwith("nvrtc_create: allocation failed");
  block = caml_alloc_custom(&nvrtc_program_ops,
    sizeof(tesserae_nvrtc_program*), 0, 1);
  *((tesserae_nvrtc_program**) Data_custom_val(block)) = prog;
  CAMLreturn(block);
}

CAMLprim value caml_nvrtc_destroy(value v) {
  CAMLparam1(v);
  tesserae_nvrtc_program* prog =
    *((tesserae_nvrtc_program**) Data_custom_val(v));
  tesserae_nvrtc_destroy(prog);
  *((tesserae_nvrtc_program**) Data_custom_val(v)) = NULL;
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nvrtc_is_valid(value v) {
  CAMLparam1(v);
  tesserae_nvrtc_program* prog =
    *((tesserae_nvrtc_program**) Data_custom_val(v));
  CAMLreturn(Val_bool(tesserae_nvrtc_is_valid(prog)));
}

CAMLprim value caml_nvrtc_compile(value v, value opts_list) {
  CAMLparam2(v, opts_list);
  CAMLlocal1(cell);
  tesserae_nvrtc_program* prog =
    *((tesserae_nvrtc_program**) Data_custom_val(v));

  int n = 0;
  cell = opts_list;
  while (cell != Val_emptylist) { n++; cell = Field(cell, 1); }

  const char** opts = malloc(n * sizeof(char*));
  cell = opts_list;
  for (int i = 0; i < n; i++) {
    opts[i] = String_val(Field(cell, 0));
    cell    = Field(cell, 1);
  }

  int rc = tesserae_nvrtc_compile(prog, opts, n);
  free(opts);

  CAMLlocal1(result);
  if (rc == 0) {
    result = caml_alloc(1, 0);
    Store_field(result, 0, Val_unit);
  } else {
    char* log = tesserae_nvrtc_get_log(prog);
    result = caml_alloc(1, 1);
    Store_field(result, 0,
      caml_copy_string(log ? log : "unknown error"));
    free(log);
  }
  CAMLreturn(result);
}

CAMLprim value caml_nvrtc_get_ptx(value v) {
  CAMLparam1(v);
  tesserae_nvrtc_program* prog =
    *((tesserae_nvrtc_program**) Data_custom_val(v));
  char* ptx = tesserae_nvrtc_get_ptx(prog);
  if (!ptx) caml_failwith("get_ptx: not compiled or failed");
  CAMLlocal1(s);
  s = caml_copy_string(ptx);
  free(ptx);
  CAMLreturn(s);
}

CAMLprim value caml_nvrtc_get_log(value v) {
  CAMLparam1(v);
  tesserae_nvrtc_program* prog =
    *((tesserae_nvrtc_program**) Data_custom_val(v));
  char* log = tesserae_nvrtc_get_log(prog);
  CAMLlocal1(s);
  s = caml_copy_string(log ? log : "");
  free(log);
  CAMLreturn(s);
}
