
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TRANSFORM_BIAS_RESCALE_QKV_H_
#define ACLNN_TRANSFORM_BIAS_RESCALE_QKV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTransformBiasRescaleQkvGetWorkspaceSize
 * parameters :
 * qkv : required
 * qkvBias : required
 * numHeads : required
 * qOut : required
 * kOut : required
 * vOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTransformBiasRescaleQkvGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qkvBias,
    int64_t numHeads,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTransformBiasRescaleQkv
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTransformBiasRescaleQkv(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
