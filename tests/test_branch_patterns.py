import csv
from pathlib import Path
from types import SimpleNamespace

import torch

from src.training.routing import update_expert_biases
from src.training.logging import log_eval_metrics, save_routing_plots
from src.utils.branch_patterns import build_branch_route_artifacts, save_branch_route_artifacts
from src.utils.load_balance_artifacts import build_load_balance_artifacts, save_load_balance_artifacts


class _FakeTokenizer:
    def decode(self, ids, clean_up_tokenization_spaces=False):
        return f"tok{int(ids[0])}"


class _FakeConfig:
    num_experts = 256
    num_attn_experts = 64


def _spiky_bias(num_experts: int, offset: float = 0.0) -> torch.Tensor:
    idx = torch.arange(num_experts, dtype=torch.float32)
    signed_blocks = torch.where((idx.remainder(7) < 3), -1.0, 1.0)
    wave = 0.55 + 0.45 * torch.sin(idx * 1.73 + 0.4).abs()
    alternating = torch.where(idx.remainder(2) == 0, 1.0, -1.0)
    return 0.01 * signed_blocks * wave + 0.0015 * alternating + offset


class _FakeRouter:
    def __init__(self, num_experts: int, offset: float = 0.0):
        self.expert_bias = _spiky_bias(num_experts, offset=offset)
        self.local_tokens_per_expert = torch.zeros(num_experts)


class _FakeAttnBank:
    routing_bundle = "qkvo"
    num_kv_experts = 64
    num_o_experts = 64

    def __init__(self):
        self.qk_routers = [
            [_FakeRouter(64, offset=0.001 * depth + 0.0001 * slot) for slot in range(2)]
            for depth in range(3)
        ]


class _FakeMlpBank:
    def __init__(self):
        self.gates = [_FakeRouter(256, offset=0.002 * depth) for depth in range(3)]


class _FakeBranchRouter:
    def __init__(self, offset: float = 0.0):
        self.expert_bias = torch.tensor([-0.02 + offset, 0.02 - offset])
        self.local_tokens_per_expert = torch.zeros(2)


class _FakeBiasOwner:
    def __init__(self, num_experts: int, counts: list[float], bias: list[float] | None = None):
        self.expert_bias = torch.tensor(bias if bias is not None else [0.0] * num_experts, dtype=torch.float32)
        self.local_tokens_per_expert = torch.tensor(counts, dtype=torch.float32)


class _FakeGlobalBiasModel:
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="moe_everything",
            global_router_update=True,
            mlp_router_balancing="deepseek_bias",
            attn_router_balancing=None,
            branch_balancing=None,
        )
        self._load_balancing_method = "deepseek_bias"
        self.r0 = _FakeBiasOwner(4, [100.0, 1.0, 1.0, 1.0], bias=[0.0, 0.0, 0.0, 0.0])
        self.r1 = _FakeBiasOwner(4, [1.0, 100.0, 1.0, 1.0], bias=[0.004, -0.002, 0.0, 0.002])

    def get_all_balancing_owners(self):
        yield self.r0, "mlp"
        yield self.r1, "mlp"


class _FakeInner:
    def __init__(self):
        self.config = _FakeConfig()
        self.branch_balancing = "deepseek_bias"
        self.attn_bank = _FakeAttnBank()
        self.mlp_bank = _FakeMlpBank()
        self.branch_routers = [_FakeBranchRouter(offset=0.001 * depth) for depth in range(3)]
        self._all_branch_selected_experts = [
            torch.tensor([[[0], [1], [0]], [[1], [1], [0]]]),
            torch.tensor([[[0], [1], [1]], [[1], [0], [0]]]),
            torch.tensor([[[1], [1], [0]], [[1], [0], [0]]]),
        ]
        self._all_branch_probs = [
            torch.tensor([
                [[0.80, 0.20], [0.30, 0.70], [0.60, 0.40]],
                [[0.45, 0.55], [0.20, 0.80], [0.75, 0.25]],
            ]),
            torch.tensor([
                [[0.70, 0.30], [0.10, 0.90], [0.40, 0.60]],
                [[0.35, 0.65], [0.65, 0.35], [0.85, 0.15]],
            ]),
            torch.tensor([
                [[0.45, 0.55], [0.20, 0.80], [0.70, 0.30]],
                [[0.25, 0.75], [0.90, 0.10], [0.80, 0.20]],
            ]),
        ]
        self._all_mlp_selected_experts = [
            torch.tensor([[10, 110], [11, 111], [12, 112], [13, 113], [14, 114], [15, 115]]),
            torch.tensor([[20, 120], [21, 121], [22, 122], [23, 123], [24, 124], [25, 125]]),
            torch.tensor([[12, 128], [31, 131], [32, 132], [33, 133], [34, 134], [35, 135]]),
        ]
        self._all_mlp_token_masks = [
            torch.tensor([False, True, False, True, True, False]),
            torch.tensor([False, True, True, True, False, False]),
            torch.tensor([True, True, False, True, False, False]),
        ]
        self._all_attn_router_info = [
            {
                "qk": {
                    "selected_experts": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]),
                    "token_mask": torch.tensor([True, False, True, False, False, True]),
                }
            },
            {
                "qk": {
                    "selected_experts": torch.tensor([[3, 4], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]]),
                    "token_mask": torch.tensor([True, False, False, False, True, True]),
                }
            },
            {
                "qk": {
                    "selected_experts": torch.tensor([[5, 6], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]]),
                    "token_mask": torch.tensor([False, False, True, False, True, True]),
                }
            },
        ]


class _FakeModel:
    def __init__(self):
        self.config = _FakeConfig()
        self.model = _FakeInner()


class _FakeFixedAlternatingModel:
    def __init__(self):
        self.config = _FakeConfig()
        self.model = _FakeInner()
        self.model.branch_balancing = "fixed_alternating"
        self.model.branch_routers = []


def test_build_branch_route_artifacts_keeps_token_patterns():
    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    payload = build_branch_route_artifacts(
        _FakeModel(),
        step=250,
        input_ids=input_ids,
        tokenizer=_FakeTokenizer(),
        top_k=4,
        max_tokens=4,
    )

    assert payload is not None
    assert payload["num_depths"] == 3
    assert payload["num_tokens"] == 6
    assert payload["sample_tokens"][0]["pattern"] == "AAM"
    assert payload["sample_tokens"][0]["token_text"] == "tok10"
    assert payload["sample_tokens"][0]["expert_pattern"] == "A1|A2 A3|A4 M12|M128"
    assert payload["depth_summary"][0]["attn_tokens"] == 3

    patterns = {row["pattern"]: row for row in payload["top_patterns"]}
    assert patterns["MMM"]["count"] == 2
    assert patterns["AAM"]["examples"][0]["position"] == 0


def test_save_branch_route_artifacts_writes_expected_files(tmp_path: Path):
    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    wrote = save_branch_route_artifacts(
        _FakeModel(),
        step_dir=str(tmp_path),
        step=250,
        input_ids=input_ids,
        tokenizer=_FakeTokenizer(),
        top_k=4,
        max_tokens=4,
    )

    out_dir = tmp_path / "branch_patterns"
    assert wrote
    assert (out_dir / "token_routes.csv").exists()
    assert (out_dir / "token_routes.md").exists()
    assert (out_dir / "top_patterns.csv").exists()
    assert (out_dir / "depth_summary.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "top_patterns.png").exists()
    assert (out_dir / "branch_depth_ratios.png").exists()
    assert (out_dir / "top_pattern_matrix.png").exists()
    assert not (out_dir / "summary.json").exists()
    assert not (out_dir / "token_routes.jsonl").exists()
    assert not (out_dir / "token_grid.txt").exists()
    table = (out_dir / "token_routes.csv").read_text()
    assert "AAM" in table
    assert "M12|M128" in table
    assert "pattern_rank" in table
    rows = list(csv.DictReader(table.splitlines()))
    assert rows[0]["layer_02"] == "M12|M128"
    assert rows[0]["layer_02_attn"] == ""
    assert rows[0]["layer_02_mlp"] == "M12|M128"
    markdown = (out_dir / "summary.md").read_text()
    assert "## Top Branch Patterns" in markdown
    assert "## Sample Token Routes" in markdown
    assert "A1&#124;A2" in markdown
    route_markdown = (out_dir / "token_routes.md").read_text()
    assert "# Token Routes" in route_markdown
    assert "Route by depth" in route_markdown


def test_load_balance_artifacts_include_mlp_attention_and_branch(tmp_path: Path):
    payload = build_load_balance_artifacts(_FakeModel(), step=250)

    assert payload is not None
    pools = set(payload["pools"])
    assert {"mlp", "attn:qkvo", "branch"}.issubset(pools)
    global_rows = {
        row["pool"]: row for row in payload["summary_rows"]
        if row["depth"] == ""
    }
    assert global_rows["mlp"]["num_experts"] == 256
    assert global_rows["attn:qkvo"]["num_experts"] == 64
    assert global_rows["branch"]["num_experts"] == 2
    assert {"mlp", "attn:qkvo", "branch"}.issubset(set(payload["bias_pools"]))

    wrote = save_load_balance_artifacts(_FakeModel(), step_dir=str(tmp_path), step=250)
    out_dir = tmp_path / "load_balancing"
    assert wrote
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "expert_load.csv").exists()
    assert (out_dir / "bias.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "summary.png").exists()
    assert (out_dir / "mlp" / "heatmap.png").exists()
    assert (out_dir / "mlp" / "global_histogram.png").exists()
    assert (out_dir / "mlp" / "per_layer_histograms.png").exists()
    assert (out_dir / "mlp" / "bias.csv").exists()
    assert (out_dir / "mlp" / "bias_by_expert.png").exists()
    assert (out_dir / "attn" / "qkvo" / "summary.md").exists()
    assert (out_dir / "attn" / "qkvo" / "heatmap.png").exists()
    assert (out_dir / "attn" / "qkvo" / "global_histogram.png").exists()
    assert (out_dir / "attn" / "qkvo" / "per_layer_histograms.png").exists()
    assert (out_dir / "attn" / "qkvo" / "bias.csv").exists()
    assert (out_dir / "attn" / "qkvo" / "bias_by_expert.png").exists()
    assert not (out_dir / "attn" / "qkvo" / "bias_per_layer_heatmap.png").exists()
    assert (out_dir / "branch" / "summary.md").exists()
    assert (out_dir / "branch" / "global_histogram.png").exists()
    assert (out_dir / "branch" / "bias.csv").exists()
    markdown = (out_dir / "summary.md").read_text()
    assert "## Global Pool Summary" in markdown
    assert "## Top Loaded Experts" in markdown
    assert "## Bias Summary" in markdown
    assert "attn:qkvo" in markdown
    assert payload["bias_pools"]["attn:qkvo"]["summary"]["num_experts"] == 64


def test_global_router_update_broadcasts_one_bias_per_pool():
    model = _FakeGlobalBiasModel()

    with torch.no_grad():
        update_expert_biases(model, bias_rate=0.01, distributed=False)

    torch.testing.assert_close(model.r0.expert_bias, model.r1.expert_bias)
    assert model.r0.expert_bias[0] < 0
    assert model.r0.expert_bias[1] < 0
    assert model.r0.expert_bias[2] > 0
    assert model.r0.expert_bias[3] > 0
    assert model.r0.local_tokens_per_expert.sum().item() == 0.0
    assert model.r1.local_tokens_per_expert.sum().item() == 0.0


def test_fixed_alternating_load_balance_skips_branch_folder(tmp_path: Path):
    payload = build_load_balance_artifacts(_FakeFixedAlternatingModel(), step=250)

    assert payload is not None
    assert "branch" not in set(payload["pools"])

    wrote = save_load_balance_artifacts(_FakeFixedAlternatingModel(), step_dir=str(tmp_path), step=250)
    out_dir = tmp_path / "load_balancing"
    assert wrote
    assert not (out_dir / "branch").exists()
    assert (out_dir / "attn" / "qkvo" / "summary.md").exists()
    assert (out_dir / "mlp" / "summary.md").exists()


def test_save_routing_plots_writes_branch_patterns_without_routing_stats(tmp_path: Path):
    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])

    save_routing_plots(
        _FakeModel(),
        output_dir=str(tmp_path),
        step=250,
        input_ids=input_ids,
        tokenizer=_FakeTokenizer(),
    )

    assert (
        tmp_path
        / "routing_logs"
        / "step_00000250"
        / "branch_patterns"
        / "token_routes.csv"
    ).exists()
    assert (
        tmp_path
        / "routing_logs"
        / "step_00000250"
        / "load_balancing"
        / "summary.csv"
    ).exists()
    assert (
        tmp_path
        / "routing_logs"
        / "step_00000250"
        / "load_balancing"
        / "summary.md"
    ).exists()
    assert (
        tmp_path
        / "routing_logs"
        / "step_00000250"
        / "load_balancing"
        / "summary.png"
    ).exists()
    assert (
        tmp_path
        / "routing_logs"
        / "step_00000250"
        / "load_balancing"
        / "attn"
        / "qkvo"
        / "summary.md"
    ).exists()


def test_log_eval_metrics_writes_local_eval_artifacts(tmp_path: Path):
    metrics = {"eval/ce_loss": 3.25, "eval/aux_loss": 0.01}

    log_eval_metrics(None, step=250, eval_metrics=metrics, output_dir=str(tmp_path))

    assert (tmp_path / "eval_logs" / "step_00000250" / "metrics.json").exists()
    history = tmp_path / "eval_logs" / "eval_metrics.jsonl"
    assert history.exists()
    assert '"eval/perplexity"' in history.read_text()
