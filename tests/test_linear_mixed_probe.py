import torch
import torch.nn as nn


def test_collect_internvl2_linear_scores_includes_attention_wo():
    from qmllm.methods.mbq.quantize.pre_quant import _collect_internvl2_linear_scores

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wqkv = nn.Linear(8, 8, bias=False)
            self.wo = nn.Linear(8, 8, bias=False)

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(8, 8, bias=False)
            self.w2 = nn.Linear(8, 8, bias=False)
            self.w3 = nn.Linear(8, 8, bias=False)

    class InternLM2DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.feed_forward = FeedForward()

    layer = InternLM2DecoderLayer()
    input_feat = {
        "attention.wqkv": torch.randn(1, 4, 8),
        "attention.wo": torch.randn(1, 4, 8),
        "feed_forward.w1": torch.randn(1, 4, 8),
        "feed_forward.w2": torch.randn(1, 4, 8),
        "feed_forward.w3": torch.randn(1, 4, 8),
    }
    ans_mask = torch.tensor([[False, True, False, True]])
    vis_mask = torch.tensor([[True, False, True, False]])

    entries = _collect_internvl2_linear_scores(
        layer=layer,
        layer_name="model.layers.0",
        input_feat=input_feat,
        w_bit=2,
        q_config={"zero_point": True, "q_group_size": -1},
        ans_mask=ans_mask,
        vis_mask=vis_mask,
        reweight_ratio_dict={"attn": 2.0, "mlp": 1.0},
    )

    assert len(entries) == 5
    assert any(item["module_type"] == "attention_wo" for item in entries)
    assert any(item["name"] == "model.layers.0.attention.wo" for item in entries)


def test_build_linear_bit_map_keeps_top_half_at_high_bit():
    from qmllm.methods.mbq.quantize.pre_quant import _build_linear_bit_map

    entries = [
        {"name": "model.layers.0.attention.wqkv", "score": 0.9},
        {"name": "model.layers.0.attention.wo", "score": 0.8},
        {"name": "model.layers.0.feed_forward.w1", "score": 0.4},
        {"name": "model.layers.0.feed_forward.w2", "score": 0.2},
    ]

    bit_map = _build_linear_bit_map(
        entries,
        default_w_bit=2,
        high_w_bit=4,
        keep_ratio=0.5,
    )

    assert bit_map["model.layers.0.attention.wqkv"] == 4
    assert bit_map["model.layers.0.attention.wo"] == 4
    assert bit_map["model.layers.0.feed_forward.w1"] == 2
    assert bit_map["model.layers.0.feed_forward.w2"] == 2


def test_pseudo_quantize_model_weight_uses_linear_bit_map(monkeypatch):
    from qmllm.methods.mbq.quantize import quantizer as quantizer_module

    class FakeLayer(nn.Module):
        def __init__(self, name):
            super().__init__()
            self._name = name
            self.linear = nn.Linear(8, 8, bias=False)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [FakeLayer("model.layers.0"), FakeLayer("model.layers.1")]
            )

    model = FakeModel()
    seen_bits = []

    def fake_quantize_tensor(tensor, n_bits=8, **kwargs):
        seen_bits.append(int(n_bits))
        return tensor.clone()

    monkeypatch.setattr(
        quantizer_module,
        "pseudo_quantize_tensor",
        fake_quantize_tensor,
    )

    import qmllm.methods.mbq.quantize.pre_quant as pre_quant_module
    import qmllm.utils.search as search_module

    monkeypatch.setattr(pre_quant_module, "get_blocks", lambda model_: model_.layers)
    monkeypatch.setattr(
        pre_quant_module,
        "get_named_linears",
        lambda layer: {"linear": layer.linear},
    )
    monkeypatch.setattr(
        search_module,
        "get_op_name",
        lambda model_, layer: layer._name,
    )

    quantizer_module.pseudo_quantize_model_weight(
        model,
        w_bit=2,
        q_config={"zero_point": True, "q_group_size": -1},
        layer_bit_map={"model.layers.0.linear": 4},
    )

    assert seen_bits == [4, 2]
