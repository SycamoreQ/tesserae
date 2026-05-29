#include <cuda.h>
#include <cuda_runtime.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/bigarray.h>
#include <stdio.h>
#include <stdlib.h>

#define CU_CHECK(rc) do { \
  CUresult _rc = (rc); \
  if (_rc != CUDA_SUCCESS) { \
    const char* msg; \
    cuGetErrorString(_rc, &msg); \
    caml_failwith(msg ? msg : "CUDA driver error"); \
  } \
} while(0)

#define CUDA_CHECK(rc) do { \
  if ((rc) != cudaSuccess) \
    caml_failwith(cudaGetErrorString(rc)); \
} while(0)

// Forward declaration matching the Bigarray tracking layout
CAMLprim value caml_launch_kernel(value fn_val, value gx, value gy, value gz,
                                  value bx, value by, value bz, value smem_val,
                                  value args_bigarray);

CAMLprim value caml_cuinit(value unit) {
  CAMLparam1(unit);
  CU_CHECK(cuInit(0));
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU_CHECK(cuCtxCreate(&ctx, NULL, 0, dev));

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

CAMLprim value caml_create_tma_descriptor(
    value ptr_val,
    value rows_val,
    value cols_val,
    value tile_rows_val,
    value tile_cols_val)
{
  CAMLparam5(ptr_val, rows_val, cols_val, tile_rows_val, tile_cols_val);

  CUtensorMap* tmap = (CUtensorMap*) aligned_alloc(64, sizeof(CUtensorMap));
  if (!tmap) caml_failwith("aligned_alloc failed for CUtensorMap");

  void* global_ptr = (void*)(intnat) Nativeint_val(ptr_val);
  uint64_t global_dim[2]    = { (uint64_t)Int_val(cols_val),
                                 (uint64_t)Int_val(rows_val) };
  uint64_t global_stride[1] = { (uint64_t)Int_val(cols_val) * sizeof(uint16_t) };
  uint32_t box_dim[2]       = { (uint32_t)Int_val(tile_cols_val),
                                 (uint32_t)Int_val(tile_rows_val) };
  uint32_t elem_stride[2]   = { 1, 1 };

  CUresult rc = cuTensorMapEncodeTiled(
    tmap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2,
    global_ptr,
    global_dim,
    global_stride,
    box_dim,
    elem_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_64B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  if (rc != CUDA_SUCCESS) {
    free(tmap);
    const char* msg;
    cuGetErrorString(rc, &msg);
    caml_failwith(msg ? msg : "cuTensorMapEncodeTiled failed");
  }

  CAMLreturn(caml_copy_nativeint((intnat) tmap));
}

CAMLprim value caml_free_tma_descriptor(value ptr_val) {
  CAMLparam1(ptr_val);
  void* p = (void*)(intnat) Nativeint_val(ptr_val);
  free(p);
  CAMLreturn(Val_unit);
}


CAMLprim value caml_launch_kernel(
    value fn_val, value gx, value gy, value gz,
    value bx, value by, value bz, value smem_val,
    value args_bigarray) {
  CAMLparam5(fn_val, gx, gy, smem_val, args_bigarray);
  CAMLxparam4(gz, bx, by, bz);

  CUfunction fn = (CUfunction) Nativeint_val(fn_val);
  int smem = Int_val(smem_val);

  /* Opt into extended shared memory — required when smem > 48KB */
  CUresult attr_rc = cuFuncSetAttribute(fn,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);
  if (attr_rc != CUDA_SUCCESS) {
    const char* msg;
    cuGetErrorString(attr_rc, &msg);
    caml_failwith(msg ? msg : "cuFuncSetAttribute failed");
  }

  struct caml_ba_array* ba = Caml_ba_array_val(args_bigarray);
  intnat n = ba->dim[0];
  intnat* data = (intnat*) ba->data;
  void** params = (void**) malloc(n * sizeof(void*));
  for (intnat i = 0; i < n; i++) params[i] = &data[i];

  CUresult rc = cuLaunchKernel(fn,
    Int_val(gx), Int_val(gy), Int_val(gz),
    Int_val(bx), Int_val(by), Int_val(bz),
    smem, NULL, params, NULL);

  free(params);
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

  char buf[512];
  snprintf(buf, sizeof(buf), "%s (sm_%d%d)",
    prop.name,
    prop.major,
    prop.minor);
  CAMLreturn(caml_copy_string(buf));
}
