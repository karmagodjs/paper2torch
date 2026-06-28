from dataclasses import dataclass

@dataclass
class Config:
    num_heads: int = 8
    num_layers: int = 6
    d_model: int = 512
    dff: int = 2048
    dropout_rate: float = 0.1
    max_positional_length: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 32
    sequence_length: int = 50
    vocab_size: int = 50000
    warmup_steps: int = 4000