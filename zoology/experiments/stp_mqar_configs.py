"""
STP-Transformer MQAR Experiment Configs for Zoology
=====================================================

Tests all STP approaches (A, B, C) and combinations against baselines
on the standard MQAR benchmark from Arora et al. (ICLR 2024).

This config matches the standard Zoology MQAR setup:
  - vocab_size: 8192
  - d_model: swept over [64, 128]
  - n_layers: 2
  - state_mixer: MLP with hidden_mult=4 (default)
  - batch_size: 256
  - lr: grid search over [1e-3, 3e-3, 1e-2, 3e-2]
  - 32 epochs over 100K training examples

Models tested:
  1. Softmax Attention (upper bound baseline)
  2. LinearAttention (no-decay baseline)
  3. RetNetAttention (scalar-decay baseline)
  4. STPAttention (Approach A: per-element decay + Hebbian)
  5. HybridSTPAttention (Approach B: softmax + STP gated)
  6. Softmax + STPMLP (Approach C: plastic FFN)
  7. STPAttention + STPMLP (A+C: full plasticity)
  8. HybridSTPAttention + STPMLP (B+C: everything)

Usage:
    # From the zoology root directory:
    python -m zoology.launch zoology/experiments/stp_mqar_configs.py
    python -m zoology.launch zoology/experiments/stp_mqar_configs.py -p  # parallel GPUs

Author: RISE Lab, Purdue University
Date: February 2026
"""

import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "stp-mqar-" + sweep_id

VOCAB_SIZE = 8_192

# ==============================================================================
# 1. Data configuration (matches standard Zoology MQAR)
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

# Common factory kwargs
no_mlp_kwargs = {
    "vocab_size": VOCAB_SIZE,
}

# Standard MLP state mixer (Zoology default)
standard_mlp = ModuleConfig(
    name="zoology.mixers.mlp.MLP",
    kwargs={"hidden_mult": 4},
)

# STP MLP state mixer (Approach C)
stp_mlp = ModuleConfig(
    name="zoology.mixers.stp.STPMLP",
    kwargs={"hidden_mult": 4},
)


# ==============================================================================
# 2. Model configurations
# ==============================================================================

models = []

for d_model in [64, 128]:

    # --- Baseline 1: Softmax Attention ---
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1},
        ),
        state_mixer=standard_mlp,
        max_position_embeddings=0,
        name="softmax_attn",
        **no_mlp_kwargs,
    ))

    # --- Baseline 2: LinearAttention ---
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.LinearAttention",
            kwargs={"num_heads": 1},
        ),
        state_mixer=standard_mlp,
        max_position_embeddings=0,
        name="linear_attn",
        **no_mlp_kwargs,
    ))

    # --- Baseline 3: RetNet ---
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.RetNetAttention",
            kwargs={"num_heads": 1},
        ),
        state_mixer=standard_mlp,
        max_position_embeddings=0,
        name="retnet_attn",
        **no_mlp_kwargs,
    ))

    # --- Approach A: STP Attention + standard MLP ---
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.STPAttention",
            kwargs={"num_heads": 1},
        ),
        state_mixer=standard_mlp,
        max_position_embeddings=0,
        name="stp_attn_A",
        **no_mlp_kwargs,
    ))

    # --- Approach B: Hybrid STP+Softmax + standard MLP ---
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.HybridSTPAttention",
            kwargs={"num_heads": 1},
        ),
        state_mixer=standard_mlp,
        max_position_embeddings=0,
        name="hybrid_stp_B",
        **no_mlp_kwargs,
    ))

    # --- Approach C: Softmax Attention + STP MLP ---
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1},
        ),
        state_mixer=stp_mlp,
        max_position_embeddings=0,
        name="stp_mlp_C",
        **no_mlp_kwargs,
    ))

    # --- Approach A+C: STP Attention + STP MLP ---
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.STPAttention",
            kwargs={"num_heads": 1},
        ),
        state_mixer=stp_mlp,
        max_position_embeddings=0,
        name="stp_full_AC",
        **no_mlp_kwargs,
    ))

    # --- Approach B+C: Hybrid + STP MLP ---
    models.append(ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.stp.HybridSTPAttention",
            kwargs={"num_heads": 1},
        ),
        state_mixer=stp_mlp,
        max_position_embeddings=0,
        name="hybrid_stp_BC",
        **no_mlp_kwargs,
    ))


# ==============================================================================
# 3. Training configs with learning rate sweep
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
                project_name="stp_mqar",
                entity="rise-lab",
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
        )
        configs.append(config)

print(f"Total configs: {len(configs)}")
print(f"  Models: {len(models)} ({len(models)//2} per d_model x 2 d_models)")
print(f"  LR values: 4")
print(f"  Total runs: {len(configs)}")
