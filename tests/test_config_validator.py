"""AC-6 validator/runtime alignment tests.

Historically the validator (`scripts/validate_configs.py`) included `bundled`
in `VALID_ATTN_EXPERT_MODES`, but the runtime
(`src/training/model_factory.py::build_model` via
`src/models/moe_everything/attention_bank.py`) rejects that value with
`ValueError`. This test pins the two layers together so a future edit of
either side surfaces the mismatch.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the validator module as a module object (it's a script, not a package).
import importlib.util
_VALIDATOR_PATH = Path(__file__).resolve().parent.parent / "scripts" / "validate_configs.py"
_spec = importlib.util.spec_from_file_location("validate_configs_module", _VALIDATOR_PATH)
validate_configs_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_configs_module)
VALID_ATTN_EXPERT_MODES = validate_configs_module.VALID_ATTN_EXPERT_MODES
validate_config = validate_configs_module.validate_config


def _write_moe_everything_config(
    path: Path,
    attn_expert_mode: str,
    attn_routing_bundle: str | None = None,
) -> None:
    bundle_line = (
        f"          attn_routing_bundle: {attn_routing_bundle}\n"
        if attn_routing_bundle is not None
        else ""
    )
    path.write_text(textwrap.dedent(f"""\
        model:
          type: moe_everything
          vocab_size: 256
          hidden_size: 64
          num_hidden_layers: 2
          head_dim: 16
          num_attention_heads: 4
          num_key_value_heads: 2
          num_experts: 4
          num_experts_per_tok: 2
          moe_intermediate_size: 32
          intermediate_size: 128
          max_position_embeddings: 128
          attn_expert_mode: {attn_expert_mode}
{bundle_line.rstrip()}
        training:
          learning_rate: 1.0e-3
          weight_decay: 0.0
          max_grad_norm: 1.0
          lr_scheduler: cosine
          warmup_steps: 0
          max_steps: 2
          batch_size: 2
          gradient_accumulation: 1
          mixed_precision: bf16
          output_dir: /tmp/out
        data:
          data_dir: /tmp/data
          text_column: text
          seq_len: 32
          tokenizer_name: gpt2
        eval:
          enabled: false
        checkpoint:
          resume_from: null
    """))


def test_valid_attn_expert_modes_match_runtime_enum():
    from src.models.moe_everything.attention_bank import ATTN_EXPERT_MODES

    runtime_modes = {"per_head_no_recompute", "per_head_recompute_k", "per_head_recompute_kv"}
    assert VALID_ATTN_EXPERT_MODES == runtime_modes, (
        "Validator enum drifted from runtime. Validator has "
        f"{VALID_ATTN_EXPERT_MODES}, runtime rejects everything outside "
        f"{runtime_modes}."
    )
    assert ATTN_EXPERT_MODES == runtime_modes


def test_validator_rejects_bundled_attn_expert_mode(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_moe_everything_config(cfg_path, attn_expert_mode="bundled")
    issues = validate_config(cfg_path)
    assert any("attn_expert_mode" in issue and "bundled" in issue for issue in issues), (
        f"Validator should reject bundled attn_expert_mode; got issues={issues}"
    )


def test_validator_accepts_per_head_no_recompute(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_moe_everything_config(cfg_path, attn_expert_mode="per_head_no_recompute")
    issues = validate_config(cfg_path)
    assert all("attn_expert_mode" not in issue for issue in issues), (
        f"Validator must accept per_head_no_recompute; got issues={issues}"
    )


def test_validator_accepts_per_head_recompute_kv(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_moe_everything_config(cfg_path, attn_expert_mode="per_head_recompute_kv")
    issues = validate_config(cfg_path)
    assert all("attn_expert_mode" not in issue for issue in issues), (
        f"Validator must accept per_head_recompute_kv; got issues={issues}"
    )


def test_validator_accepts_per_head_recompute_k(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_moe_everything_config(cfg_path, attn_expert_mode="per_head_recompute_k")
    issues = validate_config(cfg_path)
    assert all("attn_expert_mode" not in issue for issue in issues), (
        f"Validator must accept per_head_recompute_k; got issues={issues}"
    )


def test_validator_rejects_no_recompute_with_bundled_route(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_moe_everything_config(
        cfg_path,
        attn_expert_mode="per_head_no_recompute",
        attn_routing_bundle="qkvo",
    )
    issues = validate_config(cfg_path)
    assert any("attn_routing_bundle" in issue for issue in issues), issues


def test_validator_rejects_unknown_attn_expert_mode(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_moe_everything_config(cfg_path, attn_expert_mode="chunky_monkey")
    issues = validate_config(cfg_path)
    assert any("attn_expert_mode" in issue for issue in issues), (
        f"Validator should reject unknown modes; got issues={issues}"
    )


def test_validator_rejects_invalid_fsdp_sharding_strategy(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_moe_everything_config(cfg_path, attn_expert_mode="per_head_no_recompute")
    text = cfg_path.read_text()
    cfg_path.write_text(text.replace("mixed_precision: bf16\n", "mixed_precision: bf16\n  fsdp_sharding_strategy: zero3\n"))
    issues = validate_config(cfg_path)
    assert any("fsdp_sharding_strategy" in issue for issue in issues), issues
