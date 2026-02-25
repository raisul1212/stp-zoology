import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "stp-mqar-safe-" + sweep_id
VOCAB_SIZE = 8_192

train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]
input_seq_len = max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(train_configs=train_configs, test_configs=test_configs, batch_size=(batch_size, batch_size // 8), cache_dir="/tmp/zoology_cache")

model_factory_kwargs = {"state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE}
conv_mixer = dict(name="zoology.mixers.base_conv.BaseConv", kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True})

def make_model(d_model, mixer_dict, name):
    return ModelConfig(block_type="TransformerBlock", d_model=d_model, n_layers=2, sequence_mixer=ModuleConfig(name="zoology.mixers.hybrid.Hybrid", kwargs={"configs": [conv_mixer, mixer_dict]}), max_position_embeddings=0, name=name, **model_factory_kwargs)

models = []
for d_model in [64, 128]:
    models.append(make_model(d_model, dict(name="zoology.mixers.attention.MHA", kwargs={"dropout": 0.1, "num_heads": 2}), "softmax_attn"))
    models.append(make_model(d_model, dict(name="zoology.mixers.stp.STPAttention", kwargs={"num_heads": 2}), "stp_attn_A"))
    models.append(make_model(d_model, dict(name="zoology.mixers.stp.HybridSTPAttention", kwargs={"num_heads": 2}), "hybrid_stp_B"))

configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"{model.name}-d{model.d_model}-lr{lr:.1e}"
        configs.append(TrainConfig(model=model, data=data, learning_rate=lr, max_epochs=32, logger=LoggerConfig(project_name="stp_mqar_v2", entity="rise-lab"), slice_keys=["num_kv_pairs"], sweep_id=sweep_name, run_id=run_id))

print(f"Total configs: {len(configs)}")
print(f"  Models: softmax_attn, stp_attn_A, hybrid_stp_B")
print(f"  d_model: 64, 128")
print(f"  LRs: 4")
print(f"  Total: {len(configs)} runs")
