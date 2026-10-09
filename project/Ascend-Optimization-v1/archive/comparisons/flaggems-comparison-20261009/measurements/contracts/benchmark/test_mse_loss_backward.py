# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from . import base, consts


def mse_loss_backward_input_fn(shape, cur_dtype, device):
    grad_output = base.generate_tensor_input(shape, cur_dtype, device)
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    target = base.generate_tensor_input(shape, cur_dtype, device)
    yield grad_output, inp, target, {"reduction": 1}


class MseLossBackwardBenchmark(base.GenericBenchmark2DOnly):
    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    params={"reduction": 1},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        grad_output = base.generate_tensor_input(shape, case.dtype, self.device)
        inp = base.generate_tensor_input(shape, case.dtype, self.device)
        target = base.generate_tensor_input(shape, case.dtype, self.device)
        return grad_output, inp, target, 1


@pytest.mark.mse_loss_backward
def test_mse_loss_backward():
    bench = MseLossBackwardBenchmark(
        input_fn=mse_loss_backward_input_fn,
        op_name="mse_loss_backward",
        torch_op=torch.ops.aten.mse_loss_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
