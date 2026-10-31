// 安装头文件声明了未导出的C++包装；直接转发同名CANN公开API，保留错误检查。
#include <acl/acl.h>
#include <dlfcn.h>
#include <stdexcept>
namespace c10_npu { namespace acl {
aclError AclrtPeekAtLastError(aclrtLastErrLevel flag) {
    using Function=aclError(*)(aclrtLastErrLevel);
    static auto function=reinterpret_cast<Function>(dlsym(RTLD_DEFAULT,"aclrtPeekAtLastError"));
    if (!function) throw std::runtime_error("CANN lacks aclrtPeekAtLastError; error checking cannot be omitted");
    return function(flag);
}
}}
