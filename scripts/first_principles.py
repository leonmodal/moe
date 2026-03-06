"""
First principles: loss similar → logits similar? → hidden states similar?
Each model on its own GPU. No liger (breaks multi-GPU triton).
"""
import sys, torch, yaml
torch.backends.cuda.preferred_blas_library("cublaslt")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from accelerate.utils import set_seed
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data.parquet_dataset import DataConfig, StatefulParquetDataset
from src.models.load_balancing import seq_load_balancing_loss_func
from src.models import (Qwen3MoeConfig, DeepSeekStandardMoEModel, GlobalMoEConfig, DeepSeekGlobalMoEForCausalLM)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build(cfg):
    mcfg = cfg["model"]
    common = dict(
        vocab_size=mcfg["vocab_size"], hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"], head_dim=mcfg["head_dim"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        moe_intermediate_size=mcfg["moe_intermediate_size"],
        intermediate_size=mcfg.get("intermediate_size", 3072),
        max_position_embeddings=mcfg.get("max_position_embeddings", 32768),
        rope_theta=mcfg.get("rope_theta", 1e6), rms_norm_eps=mcfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=mcfg.get("tie_word_embeddings", False),
        router_aux_loss_coef=0.0, norm_topk_prob=True,
        num_experts_per_tok=mcfg["num_experts_per_tok"], output_router_logits=True,
    )
    if mcfg["type"] == "deepseek_standard_moe":
        config = Qwen3MoeConfig(num_experts=mcfg["num_experts"], **common)
        config.topk_scaling_factor = mcfg.get("topk_scaling_factor")
        config.num_groups = mcfg.get("num_groups")
        config.group_topk = mcfg.get("group_topk")
        model = DeepSeekStandardMoEModel(config)
    else:
        config = GlobalMoEConfig(num_experts=mcfg["num_experts"], **common)
        config.topk_scaling_factor = mcfg.get("topk_scaling_factor")
        config.num_groups = mcfg.get("num_groups")
        config.group_topk = mcfg.get("group_topk")
        model = DeepSeekGlobalMoEForCausalLM(config)
    return model, config


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float().unsqueeze(0),
                                                  b.flatten().float().unsqueeze(0)).item()


def compare(model_a, model_b, input_ids, labels, name_a, name_b, phase):
    dev_a, dev_b = next(model_a.parameters()).device, next(model_b.parameters()).device
    model_a.eval(); model_b.eval()

    with torch.no_grad(), torch.amp.autocast(dev_a.type, dtype=torch.bfloat16):
        out_a = model_a(input_ids=input_ids.to(dev_a), labels=labels.to(dev_a), output_router_logits=True)
        h_a = model_a.model(input_ids=input_ids.to(dev_a), output_router_logits=True).last_hidden_state
        logits_a = model_a.lm_head(h_a).float().cpu()
        h_a = h_a.float().cpu()

    with torch.no_grad(), torch.amp.autocast(dev_b.type, dtype=torch.bfloat16):
        out_b = model_b(input_ids=input_ids.to(dev_b), labels=labels.to(dev_b), output_router_logits=True)
        h_b = model_b.model(input_ids=input_ids.to(dev_b), output_router_logits=True).last_hidden_state
        logits_b = model_b.lm_head(h_b).float().cpu()
        h_b = h_b.float().cpu()

    pred_a = logits_a[:, :-1].argmax(dim=-1)
    pred_b = logits_b[:, :-1].argmax(dim=-1)
    agree = (pred_a == pred_b).float().mean().item()

    emb_a = model_a.lm_head.weight.data.float().cpu()
    emb_b = model_b.lm_head.weight.data.float().cpu()
    q_a = dict(model_a.named_parameters())["model.layers.0.self_attn.q_proj.weight"].data.float().cpu()
    q_b = dict(model_b.named_parameters())["model.layers.0.self_attn.q_proj.weight"].data.float().cpu()

    print(f"\n{'='*70}")
    print(f"  {phase}: {name_a} vs {name_b}")
    print(f"{'='*70}")
    print(f"  Loss:        {name_a}={out_a.loss.item():.6f}  {name_b}={out_b.loss.item():.6f}  diff={abs(out_a.loss.item()-out_b.loss.item()):.6f}")
    print(f"  Logits:      cos={cosine_sim(logits_a, logits_b):.6f}  rel_L2={(logits_a-logits_b).norm()/logits_a.norm():.4f}")
    print(f"  Predictions: {agree*100:.1f}% agree")
    print(f"  Hidden:      cos={cosine_sim(h_a, h_b):.6f}  rel_L2={(h_a-h_b).norm()/h_a.norm():.4f}")
    print(f"  LM head wt:  cos={cosine_sim(emb_a, emb_b):.6f}")
    print(f"  L0 q_proj:   cos={cosine_sim(q_a, q_b):.6f}")
    model_a.train(); model_b.train()


def train_steps(model, model_cfg, steps):
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token
    ds = StatefulParquetDataset(
        config=DataConfig(data_dir="./data/parquet", text_column="text",
                          seq_len=1024, tokenizer_name="Qwen/Qwen3-0.6B"),
        tokenizer=tokenizer, rank=0, world_size=1, seed=42)
    dl = DataLoader(ds, batch_size=16, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95))
    model.train()
    for step, batch in enumerate(dl):
        if step >= steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels, output_router_logits=True)
            loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step+1) % 10 == 0:
            print(f"    step {step+1}: loss={loss.item():.4f}")


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token
    ds = StatefulParquetDataset(
        config=DataConfig(data_dir="./data/parquet", text_column="text",
                          seq_len=1024, tokenizer_name="Qwen/Qwen3-0.6B"),
        tokenizer=tokenizer, rank=0, world_size=1, seed=999)
    eval_batch = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    eval_ids = eval_batch["input_ids"]
    eval_labels = eval_batch["labels"]

    # Each model on its own GPU
    set_seed(42)
    m_std, c_std = build(load_cfg("configs/scaling/xs_deepseek_standard.yaml"))
    m_std = m_std.to("cuda:0")

    set_seed(42)
    m_glb, c_glb = build(load_cfg("configs/scaling/xs_deepseek_global.yaml"))
    m_glb = m_glb.to("cuda:2")

    set_seed(123)
    m_std2, c_std2 = build(load_cfg("configs/scaling/xs_deepseek_standard.yaml"))
    m_std2 = m_std2.to("cuda:4")

    # AT INIT
    compare(m_std, m_glb, eval_ids, eval_labels, "Std42", "Glb42", "AT INIT")
    compare(m_std, m_std2, eval_ids, eval_labels, "Std42", "Std123", "AT INIT")

    # Train 50 steps
    print("\nTraining Std42 on cuda:0...")
    train_steps(m_std, c_std, 50)
    print("Training Glb42 on cuda:2...")
    train_steps(m_glb, c_glb, 50)
    print("Training Std123 on cuda:4...")
    train_steps(m_std2, c_std2, 50)

    # AFTER 50 STEPS
    compare(m_std, m_glb, eval_ids, eval_labels, "Std42", "Glb42", "AFTER 50 STEPS")
    compare(m_std, m_std2, eval_ids, eval_labels, "Std42", "Std123", "AFTER 50 STEPS")


if __name__ == "__main__":
    main()
