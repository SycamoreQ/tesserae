#include <cuda.h>
#include <cuda_runtime.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/fail.h>

#define CU_CHECK(rc) do { \
  if ((rc) != CUDA_SUCCESS) { \
    const char* msg; \
    cuGetErrorString((rc), &msg); \
    caml_failwith(msg ? msg : "CUDA driver error"); \
  } \
} while(0)

#define CUDA_CHECK(rc) do { \
  if ((rc) != cudaSuccess) \
    caml_failwith(cudaGetErrorString(rc)); \
} while(0)

CAMLprim value caml_cuinit(value unit) {
  CAMLparam1(unit);
  CU_CHECK(cuInit(0));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_module_load_ptx(value ptx) {
  CAMLparam1(ptx);
  CUmodule mod;
  CU_CHECK(cuModuleLoadData(&mod, String_val(ptx)));
  CAMLreturn(caml_copy_nativeint((intnat) mod));
}

CAMLprim value caml_module_unload(value mod_val) {
  CAMLparam1(mod_val);
  CUmodule mod = (CUmodule) Nativeint_val(mod_val);
  if (mod) cuModuleUnload(mod);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_get_function(value mod_val, value name) {
  CAMLparam2(mod_val, name);
  CUmodule  mod  = (CUmodule) Nativeint_val(mod_val);
  CUfunction fn;
  CU_CHECK(cuModuleGetFunction(&fn, mod, String_val(name)));
  CAMLreturn(caml_copy_nativeint((intnat) fn));
}

/* bytecode stub for caml_launch_kernel (9 args needs bytecode wrapper) */
CAMLprim value caml_launch_kernel_bytecode(value* argv, int argc) {
  (void) argc;
  return caml_launch_kernel(
    argv[0], argv[1], argv[2], argv[3],
    argv[4], argv[5], argv[6], argv[7], argv[8]);
}

CAMLprim value caml_launch_kernel(
    value fn_val,
    value gx, value gy, value gz,
    value bx, value by, value bz,
    value smem_val,
    value args_val) {
  CAMLparam5(fn_val, gx, gy, args_val, smem_val);
  CAMLxparam4(gz, bx, by, bz);

  CUfunction fn = (CUfunction) Nativeint_val(fn_val);

  /* build kernel params array from OCaml nativeint array */
  mlsize_t n = Wosize_val(args_val);
  void** params = (void**) malloc(n * sizeof(void*));
  intnat* vals  = (intnat*) malloc(n * sizeof(intnat));

  for (mlsize_t i = 0; i < n; i++) {
    vals[i]   = Nativeint_val(Field(args_val, i));
    params[i] = &vals[i];
  }

  CUresult rc = cuLaunchKernel(fn,
    Int_val(gx), Int_val(gy), Int_val(gz),
    Int_val(bx), Int_val(by), Int_val(bz),
    Int_val(smem_val), NULL,
    params, NULL);

  free(params);
  free(vals);
  CU_CHECK(rc);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_device_synchronize(value unit) {
  CAMLparam1(unit);
  CUDA_CHECK(cudaDeviceSynchronize());
  CAMLreturn(Val_unit);
}

CAMLprim value caml_device_info(value unit) {
  CAMLparam1(unit);
  int dev = 0;
  struct cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  char buf[256];
  snprintf(buf, sizeof(buf), "%s (sm_%d%d)",
    prop.name,
    prop.major,
    prop.minor);
  CAMLreturn(caml_copy_string(buf));
}
