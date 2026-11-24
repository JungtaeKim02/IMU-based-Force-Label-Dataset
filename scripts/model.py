import jax
import jax.numpy as jnp
import flax.linen as nn

class SinusoidalPositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 100

    @nn.compact
    def __call__(self, seq_len: int):
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
        pe = jnp.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return pe[None, :, :]

class LearnablePositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 16

    @nn.compact
    def __call__(self, seq_len: int):
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (self.max_len, self.d_model))
        return pos_embedding[None, :seq_len, :]

class GatedMultiHeadAttention(nn.Module):
    num_heads: int
    d_model: int
    dropout_rate: float = 0.1
    use_gated_attention: bool = True

    @nn.compact
    def __call__(self, x, training: bool=False):
        (batch_size, seq_len, _) = x.shape
        head_dim = self.d_model // self.num_heads
        if self.use_gated_attention:
            q_out = nn.Dense(self.d_model * 2, use_bias=True)(x)
            (query, gate_score) = jnp.split(q_out, 2, axis=-1)
        else:
            query = nn.Dense(self.d_model, use_bias=True)(x)
            gate_score = None
        key = nn.Dense(self.d_model, use_bias=True)(x)
        value = nn.Dense(self.d_model, use_bias=True)(x)
        query = query.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate, deterministic=not training)(attn_weights)
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        if self.use_gated_attention and gate_score is not None:
            attn_output = attn_output * nn.sigmoid(gate_score)
        output = nn.Dense(self.d_model, use_bias=True)(attn_output)
        return output

class TransformerEncoderLayer(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    use_gated_attention: bool = False

    @nn.compact
    def __call__(self, x, training: bool=False):
        residual = x
        x_norm = nn.LayerNorm()(x)
        if self.use_gated_attention:
            attn_out = GatedMultiHeadAttention(num_heads=self.num_heads, d_model=self.d_model, dropout_rate=self.dropout_rate, use_gated_attention=True)(x_norm, training=training)
        else:
            attn_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.d_model, dropout_rate=self.dropout_rate, deterministic=not training)(x_norm, x_norm)
        attn_out = nn.Dropout(self.dropout_rate, deterministic=not training)(attn_out)
        x = residual + attn_out
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.d_ff)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = residual + x
        return x

class ForceTransformer(nn.Module):
    hidden_dim: int = 192
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 384
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, sequence, training=False):
        x = nn.Dense(self.hidden_dim)(sequence)
        x = x + LearnablePositionalEncoding(self.hidden_dim, max_len=16)(sequence.shape[1])
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(self.hidden_dim, self.num_heads, self.d_ff, self.dropout_rate, use_gated_attention=True)(x, training)
        x = nn.LayerNorm()(x)
        pooled = jnp.mean(x, axis=1)
        f = nn.silu(nn.Dense(self.hidden_dim)(pooled))
        f = nn.Dropout(self.dropout_rate, deterministic=not training)(f)
        raw_force = nn.Dense(3)(f)
        gate = nn.sigmoid(nn.Dense(1)(f))
        return (gate * raw_force, raw_force, gate)
