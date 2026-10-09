#include <acl/acl.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

int main() {
  if (aclInit(nullptr) != ACL_SUCCESS) return 10;
  if (aclrtSetDevice(0) != ACL_SUCCESS) return 11;
  aclrtStream stream = nullptr;
  if (aclrtCreateStream(&stream) != ACL_SUCCESS) return 12;
  constexpr size_t n = 4096;
  void *src = nullptr, *dst = nullptr;
  if (aclrtMalloc(&src, n, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 13;
  if (aclrtMalloc(&dst, n, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 14;
  auto status = aclrtMemcpyAsync(dst, n, src, n, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status != ACL_SUCCESS) return 15;
  status = aclrtSynchronizeStream(stream);
  aclrtFree(src); aclrtFree(dst); aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
  std::cout << "aclrtMemcpyAsync DEVICE_TO_DEVICE: " << status << "\n";
  return status == ACL_SUCCESS ? 0 : 16;
}
