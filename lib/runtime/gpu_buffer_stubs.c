#include <cuda_runtime.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/custom.h>


static void gpu_buffer_finalize(value v) {
  void* ptr = *((void**) Data_custom_val(v));
  if (ptr) cudaFree(ptr);
}

static struct custom_operations gpu_buffer_ops = {
  "tesserae_gpu_buffer",
  gpu_buffer_finalize,
  custom_compare_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default,
  custom_compare_ext_default,
  custom_fixed_length_default
};


CAMLprim value caml_gpu_alloc(value n_bytes) {
  CAMLparam1(n_bytes);
  CAMLlocal1(block);
  void* ptr = NULL;
  cudaError_t rc = cudaMalloc(&ptr, Int_val(n_bytes));
  if (rc != cudaSuccess)
    caml_failwith(cudaGetErrorString(rc));
  cudaMemset(ptr, 0, Int_val(n_bytes));
  block = caml_alloc_custom(&gpu_buffer_ops, sizeof(void*), 0, 1);
  *((void**) Data_custom_val(block)) = ptr;
  CAMLreturn(block);
}


CAMLprim value caml_gpu_free(value v) {
  CAMLparam1(v);
  void* ptr = *((void**) Data_custom_val(v));
  if (ptr) {
    cudaFree(ptr);
    *((void**) Data_custom_val(v)) = NULL;
  }
  CAMLreturn(Val_unit);
}


CAMLprim value caml_gpu_copy_to_device(value v, value src, value n_bytes) {
  CAMLparam3(v, src, n_bytes);
  void* ptr = *((void**) Data_custom_val(v));
  cudaError_t rc = cudaMemcpy(
    ptr,
    (void*) Bytes_val(src),
    Int_val(n_bytes),
    cudaMemcpyHostToDevice);
  if (rc != cudaSuccess)
    caml_failwith(cudaGetErrorString(rc));
  CAMLreturn(Val_unit);
}


CAMLprim value caml_gpu_copy_to_host(value dst, value v, value n_bytes) {
  CAMLparam3(dst, v, n_bytes);
  void* ptr = *((void**) Data_custom_val(v));
  cudaError_t rc = cudaMemcpy(
    (void*) Bytes_val(dst),
    ptr,
    Int_val(n_bytes),
    cudaMemcpyDeviceToHost);
  if (rc != cudaSuccess)
    caml_failwith(cudaGetErrorString(rc));
  CAMLreturn(Val_unit);
}


CAMLprim value caml_gpu_ptr(value v) {
  CAMLparam1(v);
  void* ptr = *((void**) Data_custom_val(v));
  CAMLreturn(caml_copy_nativeint((intnat) ptr));
}
