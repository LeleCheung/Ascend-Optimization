"""沿用固定上游的有效前缀语义，逐次读取本用例的masked_m。"""
def valid(ref_outputs, sol_outputs, inputs, ctx):
    try:
        expected, actual = ref_outputs[0], sol_outputs[0]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        counts = inputs['masked_m'].tolist()
        assert len(counts) == expected.shape[0]
        for expert, count in enumerate(counts):
            if count <= 0:
                continue
            # 上游不约束有效前缀之后的填充区；不将其擅自改成必须为零。
            assert_close(actual[expert, :int(count)], expected[expert, :int(count)])
    except (AssertionError, IndexError, TypeError, ValueError) as error:
        return {'passed': False, 'message': str(error), 'metrics': {}}
    return {'passed': True, 'message': '', 'metrics': {}}
