from dataclasses import dataclass

@dataclass
class Config:
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    d_inner: int = 2048
    dropout: float = 0.1
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    batch_size: int = 25000
    max_sequence_length: int = 100
    position_encoding_max_length: int = 500
    optimizer: str = 'adam'
    beta_1: float = 0.9
    beta_2: float = 0.98
    epsilon: float = 1e-9
    beam_size: int = 4
    length_penalty: float = 0.6