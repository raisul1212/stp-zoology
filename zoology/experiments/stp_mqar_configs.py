"""
STP-Transformer MQAR Experiment Configs for Zoology — v2
=========================================================

Clean apples-to-apples comparison. ALL models share the EXACT same
architecture from the original Zoology MQAR benchmark:
  - TransformerBlock
  - Layer 0: BaseConv (kernel_size=3)
  - Layer 1: [THE MIXER BEING TESTED]
  - state_mixer: Identity (NO MLP)
  - n_layers: 2
  - d_model: swept over [64, 128]
  - lr: swept over [1e-3, 3.2e-3, 1e-2, 3.2e-2]

The ONLY difference between models is the sequence mixer in Layer 1.

Baselines (Zoology's own implementations):
  1. softmax_attn — Softmax MHA
  2. based       — Based linear attention (Taylor exp feature map)
  3. delta_net   — DeltaNet (delta rule fast weight programmer)
  4. gla         — Gated Linear Attention (per-row decay)

STP variants (our new implementations):
  5. stp_attn_A  — STP Attention (Approach A: per-element decay + Hebbian)
  6. hybrid_stp_B — Hybrid Softmax+STP with learned gate (Approach B)

Total: 6 models x 2 d_models x 4 LRs = 48 runs

Usage:
    export WANDB_MODE=offline
    python -m zoology.launch zoology/experiments/stp_mqar_configs.py

Author: RISE Lab, Purdue University
Date: February 2026
"""

import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "stp-mqar-v2-" + sweep_id

VOCAB_SIZE = 8_192

# ==============================================================================
# 1. Data configuration (identical to original Zoology MQAR)
# ==============================================================================

train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024,num_examples=1_000, num_kv_pairs=256),
]

input_seq_len = max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(batch_size, batch_size // 8),
    cache_dir="/tmp/zoology_cache",
)

# ==============================================================================
# 2. Shared components (matching original config EXACTLY)
# ==============================================================================

# Original uses state_mixer=Identity (NO MLP)
model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# Original uses conv_mixer as layer 0
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)

# Helper: create model with [conv, mixer] hybrid pattern
def make_model(d_model, mixer_dict, name):
    return ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, mixer_dict]},
        ),
        max_position_embeddings=0,
        name=name,
        **model_factory_kwargs,
    )


# ==============================================================================
# 3. Model configurations
# ==============================================================================

models = []

for d_model in [64, 128]:

    # ==========================================
    # BASELINES — Zoology's own implementations
    # ==========================================

    # Softmax Attention
    models.append(make_model(d_model, dict(
        name="zoology.mixers.attention.MHA",
        kwargs={"dropout": 0.1, "num_heads": 2},
    ), "softmax_attn"))

    # Based (linear attention with Taylor expansion feature map)
    models.append(make_model(d_model, dict(
        name="zoology.mixers.based.Based",
        kwargs={
            "l_max": input_seq_len,
            "feature_dim": 16,
            "feature_name": "taylor_exp",
            "num_key_value_heads": 1,
            "num_heads": 1,
            "train_view": "quadratic",
        },
    ), "based"))

    # DeltaNet (delta rule fast weight programmer — closest to STP)
    models.append(make_model(d_model, dict(
        name="zoology.mixers.delta_net.DeltaNet",
        kwargs={
            "l_max": input_seq_len,
            "num_heads": 2,
            "use_beta": True,
            "use_gate": False,
            "use_short_conv": True,
            "conv_size": 4,
        },
    ), "delta_net"))

    # GLA (Gated Linear Attention — per-row decay)
    models.append(make_model(d_model, dict(
        name="zoology.mixers.gla.GatedLinearAttention",
        kwargs={
            "num_heads": 2,
            "use_short_conv": False,
        },
    ), "gla"))

    # ==========================================
    # STP VARIANTS — our new implementations
    # ==========================================

    # Approach A: STP Attention (per-element decay + Hebbian learning)
    models.append(make_model(d_model, dict(
        name="zoology.mixers.stp.STPAttention",
        kwargs={"num_heads": 2},
    ), "stp_attn_A"))

    # Approach B: Hybrid Softmax + STP with learned gate
    models.append(make_model(d_model, dict(
        name="zoology.mixers.stp.HybridSTPAttention",
        kwargs={"num_heads": 2},
    ), "hybrid_stp_B"))


# ==============================================================================
# 4. Training configs with learning rate sweep
# ==============================================================================

configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"{model.name}-d{model.d_model}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name="stp_mqar_v2",
                entity="rise-lab",
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
        )
        configs.append(config)

print(f"Total configs: {len(configs)}")
print(f"  Models per d_model: {len(models)//2}")
print(f"    Baselines: softmax_attn, based, delta_net, gla")
print(f"    STP variants: stp_attn_A, hybrid_stp_B")
print(f"  d_model values: 2 (64, 128)")
print(f"  LR values: 4")
print(f"  Total runs: {len(configs)}")
