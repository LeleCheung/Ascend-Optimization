
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FOREACH_EXPM1_H_
#define ACLNN_FOREACH_EXPM1_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnForeachExpm1GetWorkspaceSize
 * parameters :
 * x : dynamic
 * out : dynamic
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachExpm1GetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnForeachExpm1
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachExpm1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
