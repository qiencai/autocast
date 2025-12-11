from azula.nn.embedding import SineEncoding
from azula.nn.unet import UNet
from einops import rearrange
from torch import nn

from autocast.types import Tensor, TensorBTSC


class TemporalAttention(nn.Module):
    """Temporal self-attention that processes the time dimension."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Project to higher dimension if needed
        if hidden_dim is None:
            # Ensure divisible by num_heads
            hidden_dim = max(
                channels,
                num_heads * ((channels + num_heads - 1) // num_heads)
            )

        self.hidden_dim = hidden_dim

        # Project input to hidden dimension
        self.proj_in = (
            nn.Linear(channels, hidden_dim)
            if channels != hidden_dim
            else nn.Identity()
        )

        # Use GroupNorm for spatial dimensions
        self.norm = nn.GroupNorm(min(8, hidden_dim), hidden_dim)

        # Multi-head attention over temporal dimension
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Project back to original dimension
        self.proj_out = (
            nn.Linear(hidden_dim, channels)
            if channels != hidden_dim
            else nn.Identity()
        )

    def forward(self, x: TensorBTSC) -> TensorBTSC:
        """Apply temporal self-attention.

        Args:
            x: (B, T, H, W, C)

        Returns
        -------
            (B, T, H, W, C) with temporal attention applied
        """
        B, _, H, W, _ = x.shape

        # Project to hidden dimension: (B, T, H, W, C) -> (B, T, H, W, hidden_dim)
        x_proj = self.proj_in(x)

        # Normalize spatially
        x_norm = rearrange(x_proj, "b t h w c -> b c t h w")
        x_norm = self.norm(x_norm)
        x_norm = rearrange(x_norm, "b c t h w -> b t h w c")

        # Reshape to (B*H*W, T, hidden_dim) for temporal attention
        x_flat = rearrange(x_norm, "b t h w c -> (b h w) t c")

        # Apply temporal self-attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)

        # Reshape back
        attn_out = rearrange(attn_out, "(b h w) t c -> b t h w c", b=B, h=H, w=W)

        # Project back to original dimension
        attn_out = self.proj_out(attn_out)

        # Residual connection
        return x + attn_out


class TemporalUNetBackbone(nn.Module):
    """Azula UNet with proper time embedding and temporal attention."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        n_steps_output: int = 4,
        n_steps_input: int = 1,
        mod_features: int = 256,
        hid_channels: tuple = (32, 64, 128),
        hid_blocks: tuple = (2, 2, 2),
        spatial: int = 2,
        periodic: bool = False,
        use_temporal_attention: bool = True,
        num_attention_heads: int = 8,
        attention_hidden_dim: int = 64,
    ):
        super().__init__()

        self.use_temporal_attention = use_temporal_attention
        self.n_steps_output = n_steps_output
        self.n_steps_input = n_steps_input

        # Time embedding for diffusion timestep
        self.time_embedding = nn.Sequential(
            SineEncoding(mod_features),
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, mod_features),
        )

        # Temporal attention before UNet processing
        if use_temporal_attention:
            self.temporal_attn_input = TemporalAttention(
                channels=in_channels,
                num_heads=num_attention_heads,
                hidden_dim=attention_hidden_dim
            )
            self.temporal_attn_cond = TemporalAttention(
                channels=cond_channels,
                num_heads=num_attention_heads,
                hidden_dim=attention_hidden_dim
            )

        # Azula UNet - channels are multiplied by timesteps after flattening
        self.unet = UNet(
            in_channels=in_channels * n_steps_output,
            out_channels=out_channels * n_steps_output,
            cond_channels=cond_channels * n_steps_input,
            mod_features=mod_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=3,
            stride=2,
            spatial=spatial,
            periodic=periodic,
        )

    def forward(self, x_t: TensorBTSC, t: Tensor, cond: TensorBTSC) -> TensorBTSC:
        """Forward pass of the Temporal UNet.

        Args:
            x_t: Noisy data (B, T, H, W, C) - spatial dims before channels
            t: Diffusion time steps (B,)
            cond: Conditioning input (B, T_cond, H, W, C)

        Returns
        -------
            Denoised output (B, T, H, W, C)
        """
        _, T_out, _, _, C = x_t.shape

        # Embed diffusion timestep
        t_emb = self.time_embedding(t)

        # Apply temporal attention if enabled
        if self.use_temporal_attention:
            x_t_temporal = self.temporal_attn_input(x_t)
            cond_temporal = self.temporal_attn_cond(cond)
        else:
            x_t_temporal = x_t
            cond_temporal = cond

        # Rearrange from (B, T, H, W, C) to (B, T*C, H, W) for UNet
        x_t_cf = rearrange(x_t_temporal, "b t h w c -> b (t c) h w")
        x_cond_cf = rearrange(cond_temporal, "b t h w c -> b (t c) h w")

        # UNet forward: (B, T*C, H, W) -> (B, T*out_channels, H, W)
        output = self.unet(x=x_t_cf, mod=t_emb, cond=x_cond_cf)
        # Rearrange back to (B, T, H, W, C)
        return rearrange(output, "b (t c) h w -> b t h w c", t=T_out, c=C)
