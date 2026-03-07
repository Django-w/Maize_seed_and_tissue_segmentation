import inspect
from typing import Any, Dict, Sequence, Union
import torch.nn as nn
from nnformer.network_architecture.nnFormer_synapse import nnFormer

def _filter_kwargs_by_signature(cls_or_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that appear in the callable signature (to avoid unexpected keyword errors)."""
    sig = inspect.signature(cls_or_fn)
    allowed = set(sig.parameters.keys())
    # If **kwargs is present, pass everything through
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    return {k: v for k, v in kwargs.items() if k in allowed}


def create_nnformer_model(
    in_channels: int = 1,
    out_channels: int = 4,
    img_size: Union[Sequence[int], int] = (96, 96, 96),
    **model_config,
) -> nn.Module:
    """
    Create nnFormer network as a pure nn.Module baseline.

    Requirement:
      - Install nnFormer repo as a package, e.g. pip install -e .
        (The official repo provides `nnformer/network_architecture/*.py` files.)  :contentReference[oaicite:4]{index=4}
    """
    # Choose one of the architecture files.
    # You can switch to nnFormer_acdc / nnFormer_tumor if you prefer.
  # type: ignore

    # Build kwargs (best-effort)
    kwargs = dict(model_config)
    kwargs.update(
        {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "img_size": img_size,
        }
    )

    # Filter by real signature to avoid "unexpected keyword argument"
    filtered = _filter_kwargs_by_signature(nnFormer, kwargs)

    model = nnFormer(**filtered)
    return model


if __name__ == "__main__":
    import torch

    model = create_nnformer_model(
        in_channels=1,
        out_channels=4,
        img_size=(96, 96, 96),
        channels=[16, 32, 64, 128, 256],
        depths=[2, 2, 2, 2],
        num_heads=[2, 4, 8, 16],
        patch_size=[2, 2, 2],
        window_size=[7, 7, 7],
        dropout_rate=0.0,
        norm_name="instance",
    )

    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
