#ifndef TESSERAE_NVRTC_STUBS_H
#define TESSERAE_NVRTC_STUBS_H

#include <stddef.h>

typedef struct tesserae_nvrtc_program {
  char*  source;      /* copy of the source string */
  char*  name;        /* copy of the filename*/
  char*  ptx;         /* PTX output — NULL until compiled */
  char*  log;         /* compiler log — NULL until compiled */
  int    compiled;    /* 1 if compile was called, 0 otherwise */
  int    valid;       /* 1 if not yet destroyed */

#ifdef TESSERAE_HAVE_NVRTC
  nvrtcProgram handle; /* real nvrtc handle */
#else
  void*  handle;       /* stub — unused without nvrtc */
#endif
} tesserae_nvrtc_program;


/* Create a program from source. Returns NULL on allocation failure. */
tesserae_nvrtc_program* tesserae_nvrtc_create(
  const char* source,
  const char* name);

/* Destroy a program. Safe to call multiple times (idempotent). */
void tesserae_nvrtc_destroy(tesserae_nvrtc_program* prog);

/* Returns 1 if the program handle is still valid. */
int tesserae_nvrtc_is_valid(tesserae_nvrtc_program* prog);

/* Compile the program with the given options.
   opts is an array of n_opts C strings (e.g. "--gpu-architecture=sm_80").
   Returns 0 on success, nonzero on compile error. */
int tesserae_nvrtc_compile(
  tesserae_nvrtc_program* prog,
  const char** opts,
  int n_opts);

/* Get PTX string after successful compile.
   Returns NULL if not yet compiled or compile failed.
   Caller must free() the returned string. */
char* tesserae_nvrtc_get_ptx(tesserae_nvrtc_program* prog);

/* Get compiler log (errors/warnings).
   Returns NULL if compile was never called.
   Caller must free() the returned string. */
char* tesserae_nvrtc_get_log(tesserae_nvrtc_program* prog);

/* Get the source string passed to create. Do not free. */
const char* tesserae_nvrtc_get_source(tesserae_nvrtc_program* prog);

/* Get the name string passed to create. Do not free. */
const char* tesserae_nvrtc_get_name(tesserae_nvrtc_program* prog);


#include <caml/mlvalues.h>

CAMLprim value caml_nvrtc_create(value source, value name);
CAMLprim value caml_nvrtc_destroy(value prog);
CAMLprim value caml_nvrtc_is_valid(value prog);
CAMLprim value caml_nvrtc_compile(value prog, value opts);
CAMLprim value caml_nvrtc_get_ptx(value prog);
CAMLprim value caml_nvrtc_get_log(value prog);

#endif
