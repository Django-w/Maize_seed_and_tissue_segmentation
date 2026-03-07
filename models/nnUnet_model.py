import torch.nn as nn
from typing import Sequence, Tuple, Union, Optional
from monai.networks.nets import DynUNet
import torch

def create_nnunet_model(
    in_channels: int = 1,
    out_channels: int = 4,
    img_size: Union[Sequence[int], int] = (96, 96, 96),

    # ---- compatibility layer ----
    channels: Optional[Sequence[int]] = None,
    filters: Optional[Sequence[int]] = None,
    feature_sizes: Optional[Sequence[int]] = None,

    strides: Sequence[Union[Sequence[int], int]] = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    kernel_size: Sequence[Union[Sequence[int], int]] = ((3, 3, 3),) * 5,
    upsample_kernel_size: Sequence[Union[Sequence[int], int]] = ((2, 2, 2),) * 4,

    norm_name: str = "instance",
    deep_supervision: bool = False,
) -> nn.Module:
    """
    nnU-Net-style baseline implemented via MONAI DynUNet.

    Notes:
    - Your config may use 'channels' like [32, 64, 128, 256, 320]. We map it to DynUNet 'filters'.
    - DynUNet expects:
        len(filters) == len(strides)
        len(upsample_kernel_size) == len(strides) - 1
        len(kernel_size) == len(strides)
    """
    from monai.networks.nets import DynUNet

    # 1) resolve filters priority: channels > filters > feature_sizes > default
    if channels is not None:
        resolved_filters = tuple(int(c) for c in channels)
    elif filters is not None:
        resolved_filters = tuple(int(c) for c in filters)
    elif feature_sizes is not None:
        resolved_filters = tuple(int(c) for c in feature_sizes)
    else:
        resolved_filters = (32, 64, 128, 256, 320)

    # 2) ensure tuple-of-tuples for monai
    resolved_strides = tuple(tuple(s) if isinstance(s, (list, tuple)) else (s, s, s) for s in strides)
    resolved_kernel = tuple(tuple(k) if isinstance(k, (list, tuple)) else (k, k, k) for k in kernel_size)
    resolved_up = tuple(tuple(u) if isinstance(u, (list, tuple)) else (u, u, u) for u in upsample_kernel_size)

    model = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=resolved_kernel,
        strides=resolved_strides,
        upsample_kernel_size=resolved_up,
        filters=resolved_filters,
        norm_name=norm_name,
        deep_supervision=deep_supervision,
    )
    return model


if __name__ == "__main__":
    # 测试模型创建
    model = create_nnunet_model(
        in_channels=1,
        out_channels=4,
        channels=[16, 32, 64, 128, 256],
        strides=[1, 2, 2, 2, 2],
    )

    # 测试前向传播
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("UNet 3D模型创建成功！")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")