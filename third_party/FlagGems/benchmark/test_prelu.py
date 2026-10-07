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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


class PreluBenchmark(base.Benchmark):
    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(shape={"input": shape}, params={}, builder_args=(shape,)),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        x = utils.generate_tensor_input(shape, case.dtype, self.device)
        w = torch.randn(() if len(shape) == 1 else (shape[1],), dtype=case.dtype, device=self.device)
        return x, w

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            x = utils.generate_tensor_input(shape, dtype, self.device)
            if len(shape) == 1:
                w = torch.randn((), dtype=dtype, device=self.device)
            else:
                w = torch.randn((shape[1],), dtype=dtype, device=self.device)
            yield x, w


@pytest.mark.prelu
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_prelu():
    bench = PreluBenchmark(
        op_name="prelu",
        torch_op=torch.ops.aten.prelu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
