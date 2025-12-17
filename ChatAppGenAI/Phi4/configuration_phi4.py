# configuration_phi4.py
from typing import Optional, Dict, Any
from transformers import PretrainedConfig

# This config class is a lightweight, compatible PretrainedConfig for your Phi-4 mini instruct export.
# It mirrors the hyperparameters you provided and supports from_pretrained / to_json_file usage.
# Save this file next to your model files so transformers' trust_remote_code or local imports can find it.

class Phi4Config(PretrainedConfig):
    model_type = "phi3"  # keep "phi3" to match the repo's auto_map if needed

    def __init__(
        self,
        vocab_size: int = 200064,
        hidden_size: int = 3072,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 24,
        num_key_value_heads: Optional[int] = 8,
        intermediate_size: int = 8192,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-05,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        bos_token_id: int = 199999,
        eos_token_id: int = 199999,
        pad_token_id: int = 199999,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        torch_dtype: Optional[str] = "bfloat16",
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 0.75,
        sliding_window: int = 262144,
        attention_dropout: float = 0.0,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        initializer_range: float = 0.02,
        lm_head_bias: bool = False,
        mlp_bias: bool = False,
        full_attn_mod: int = 1,
        interpolate_factor: int = 1,
        attention_bias: bool = False,
        transformer_version: Optional[str] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Core architecture
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps

        # Positional / rotary
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_scaling = rope_scaling or {}

        # Tokens and behavior
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache

        # Dropouts / misc
        self.attention_dropout = attention_dropout
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.initializer_range = initializer_range
        self.lm_head_bias = lm_head_bias
        self.mlp_bias = mlp_bias
        self.full_attn_mod = full_attn_mod
        self.interpolate_factor = interpolate_factor
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.torch_dtype = torch_dtype
        self.transformers_version = transformer_version

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        # allow loading from a config.json produced by the model repo
        return cls(**config_dict, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        # ensure serialization of all fields
        output = super().to_dict()
        # add our fields explicitly to ensure compatibility
        output.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "intermediate_size": self.intermediate_size,
                "hidden_act": self.hidden_act,
                "rms_norm_eps": self.rms_norm_eps,
                "max_position_embeddings": self.max_position_embeddings,
                "original_max_position_embeddings": self.original_max_position_embeddings,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "tie_word_embeddings": self.tie_word_embeddings,
                "use_cache": self.use_cache,
                "attention_dropout": self.attention_dropout,
                "embd_pdrop": self.embd_pdrop,
                "resid_pdrop": self.resid_pdrop,
                "initializer_range": self.initializer_range,
                "lm_head_bias": self.lm_head_bias,
                "mlp_bias": self.mlp_bias,
                "full_attn_mod": self.full_attn_mod,
                "interpolate_factor": self.interpolate_factor,
                "attention_bias": self.attention_bias,
                "sliding_window": self.sliding_window,
                "torch_dtype": self.torch_dtype,
                "transformers_version": self.transformers_version,
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_scaling": self.rope_scaling,
            }
        )
        return output