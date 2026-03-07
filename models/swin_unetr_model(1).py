"""
Swin UNETR 模型定义
使用MONAI的SwinUNETR实现
"""
import os
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from collections import OrderedDict

def create_swin_unetr_model(
    in_channels: int = 1,
    out_channels: int = 4,
    feature_size: int = 48,
    num_heads: tuple = (3, 6, 12, 24),
    depths: tuple = (2, 2, 2, 2),
    window_size: int = 7,
    patch_size: int = 2,
    qkv_bias: bool = True,
    mlp_ratio: float = 4.0,
    norm_name: str = "instance",
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
    use_checkpoint: bool = False,
    spatial_dims: int = 3,
) -> SwinUNETR:
    """
    创建Swin UNETR模型
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（类别数）
        feature_size: 特征尺寸（默认48，可调整为24或96）
        num_heads: 各层的注意力头数，元组格式 (默认: (3, 6, 12, 24))
        depths: 各层的深度，元组格式 (默认: (2, 2, 2, 2))
        window_size: Swin Transformer的窗口大小 (默认: 7)
        patch_size: 图像patch大小 (默认: 2)
        qkv_bias: 是否使用QKV偏置 (默认: True)
        mlp_ratio: MLP维度比例 (默认: 4.0)
        norm_name: 归一化类型 (默认: "instance")
        drop_rate: Dropout率 (默认: 0.0)
        attn_drop_rate: 注意力Dropout率 (默认: 0.0)
        dropout_path_rate: 路径Dropout率 (默认: 0.0)
        use_checkpoint: 是否使用梯度检查点（节省内存）(默认: False)
        spatial_dims: 空间维度，3D为3 (默认: 3)
        
    Returns:
        SwinUNETR模型
    """
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        num_heads=num_heads,
        depths=depths,
        window_size=window_size,
        patch_size=patch_size,
        qkv_bias=qkv_bias,
        mlp_ratio=mlp_ratio,
        norm_name=norm_name,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
    )
    
    return model

def load_pretrained_weights_for_pt(
    model,
    pretrained_path: str = None,
    *,
    verbose: bool = True,
    prefix_mode: str = "auto",          # "auto" | "manual"
    add_prefix: str = None,             # manual: ¸ø ckpt key Í³Ò»¼ÓÇ°×º£¨Èç "swinViT."£©
    strip_prefix: str = None,           # manual: ´Ó ckpt key Í³Ò»È¥Ç°×º£¨Èç "swinViT." / "backbone."£©
    adapt_patch_embed: bool = True,     # ÊÇ·ñ³¢ÊÔÊÊÅä patch_embed.proj.weight µÄÊäÈëÍ¨µÀ
    candidates_add_prefix = ("", "swinViT.", "backbone.", "encoder."),
    candidates_strip_prefix = ("", "module.", "model.", "net.", "network.", "encoder.", "backbone.", "swinViT.", "swinvit."),
):
    """
    Robust pretrained loader:
    - Ö§³Ö checkpoint ½á¹¹: {'state_dict':...} / {'model_state_dict':...} / Ö±½Ó¾ÍÊÇ state_dict
    - ×Ô¶¯/ÊÖ¶¯´¦Àí key Ç°×º²»Ò»ÖÂ
    - Ö»¼ÓÔØ shape ÍêÈ«Ò»ÖÂµÄ²ÎÊý£¨¿ÉÑ¡ÊÊÅä patch_embed.proj.weight µÄÊäÈëÍ¨µÀ£©
    """

    if not pretrained_path or not os.path.exists(pretrained_path):
        print("Pretrained weights file not found")
        return

    print(f"Loading pretrained weights: {pretrained_path}")
    # ¼æÈÝ²»Í¬ torch °æ±¾£ºÓÐµÄÖ§³Ö weights_only£¬ÓÐµÄ²»Ö§³Ö
    try:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(pretrained_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if verbose:
            print("Checkpoint keys:", checkpoint.keys())
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, (dict, OrderedDict)):
        print("Error: cannot find a valid state_dict in checkpoint.")
        return

    # --- 1) ÏÈ°Ñ module. Ö®ÀàµÄ³£¼ûÇ°×ºÈ¥µô£¨²»Ó°ÏìºóÃæµÄ auto/manual£© ---
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    model_sd = model.state_dict()

    def _transform_key(k: str, sp: str, ap: str) -> str:
        # strip prefix
        if sp and k.startswith(sp):
            k = k[len(sp):]
        # add prefix£¨±ÜÃâÖØ¸´¼Ó£©
        if ap and not k.startswith(ap):
            k = ap + k
        return k

    def _score_mapping(sp: str, ap: str):
        # score = ÐÎ×´Æ¥ÅäÊýÁ¿£¨×îÖØÒª£©£¬Æä´ÎÊÇ key ÃüÖÐÊýÁ¿
        hit = 0
        shape_ok = 0
        for k, v in cleaned.items():
            nk = _transform_key(k, sp, ap)
            if nk in model_sd:
                hit += 1
                if isinstance(v, torch.Tensor) and v.shape == model_sd[nk].shape:
                    shape_ok += 1
        return shape_ok, hit

    # --- 2) auto/manual ---
    if prefix_mode == "manual":
        sp = strip_prefix or ""
        ap = add_prefix or ""
        best_sp, best_ap = sp, ap
        if verbose:
            s_ok, s_hit = _score_mapping(best_sp, best_ap)
            print(f"[prefix/manual] strip='{best_sp}' add='{best_ap}'  shape_ok={s_ok} hit={s_hit}")
    else:
        # auto£ºÃ¶¾Ù strip/add µÄ×éºÏ£¬Ñ¡ shape_ok ×î´óµÄ
        best_sp, best_ap = "", ""
        best_shape_ok, best_hit = -1, -1
        for sp in candidates_strip_prefix:
            for ap in candidates_add_prefix:
                shape_ok, hit = _score_mapping(sp, ap)
                if (shape_ok > best_shape_ok) or (shape_ok == best_shape_ok and hit > best_hit):
                    best_shape_ok, best_hit = shape_ok, hit
                    best_sp, best_ap = sp, ap
        if verbose:
            print(f"[prefix/auto] best strip='{best_sp}' add='{best_ap}'  shape_ok={best_shape_ok} hit={best_hit}")

    # --- 3) patch_embed.proj.weight ---
    filtered = OrderedDict()
    skipped = 0

    for k, v in cleaned.items():
        nk = _transform_key(k, best_sp, best_ap)

        if nk not in model_sd or not isinstance(v, torch.Tensor):
            skipped += 1
            if verbose:
                print(f"Skipping layer: {k} -> {nk} (Not in model or not a Tensor)")
            continue

        # 3.1 shape Ò»ÖÂ£ºÖ±½Ó¼ÓÔØ
        if v.shape == model_sd[nk].shape:
            filtered[nk] = v
            continue

        # 3.2 ½ö¶Ô patch_embed.proj.weight ×öÊäÈëÍ¨µÀÊÊÅä£¨¿ÉÑ¡£©
        if adapt_patch_embed and nk.endswith("patch_embed.proj.weight") and v.ndim == 5 and model_sd[nk].ndim == 5:
            # v: [out, in, kD, kH, kW]
            out_c_t, in_c_t = model_sd[nk].shape[0], model_sd[nk].shape[1]
            out_c_s, in_c_s = v.shape[0], v.shape[1]

            if out_c_s == out_c_t and v.shape[2:] == model_sd[nk].shape[2:]:
                if in_c_s == 1 and in_c_t > 1:
                    # 1 -> N£ºrepeat ÔÙÆ½¾ù£¬±£³Ö³ß¶È
                    vv = v.repeat(1, in_c_t, 1, 1, 1) / float(in_c_t)
                    filtered[nk] = vv
                    if verbose:
                        print(f"[adapt] {nk}: in_channels {in_c_s} -> {in_c_t} (repeat/avg)")
                    continue
                if in_c_s > 1 and in_c_t == 1:
                    # N -> 1£º¶ÔÊäÈëÍ¨µÀ×öÆ½¾ù
                    vv = v.mean(dim=1, keepdim=True)
                    filtered[nk] = vv
                    if verbose:
                        print(f"[adapt] {nk}: in_channels {in_c_s} -> {in_c_t} (mean)")
                    continue

        skipped += 1
        if verbose:
            print(f"Skipping layer: {k} -> {nk} (shape mismatch {tuple(v.shape)} vs {tuple(model_sd[nk].shape)})")

    # --- 4) ¼ÓÔØ²¢»ã×Ü ---
    incompatible = model.load_state_dict(filtered, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])

    print(f"Loaded {len(filtered)} matching weight tensors (skipped {skipped})")
    if verbose:
        print(f"Missing keys (example, first 20): {missing[:20]}")
        print(f"Unexpected keys (example, first 20): {unexpected[:20]}")


def load_pretrained_weights(model: SwinUNETR, pretrained_path: str = None):
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # Print the keys in the checkpoint to check the structure
        print("Checkpoint keys:", checkpoint.keys())

        # Get the model's state_dict from the checkpoint, checking for different possible keys
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

        if state_dict is None:
            print("Error: 'model_state_dict' not found in checkpoint.")
            return

        model_state_dict = model.state_dict()
        filtered_state_dict = {}

        # If the checkpoint has "module" prefix, strip it from the keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v

        # Load the filtered weights
        for k, v in new_state_dict.items():
            if k in model_state_dict and isinstance(v, torch.Tensor) and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping layer: {k} (Not in model or shape mismatch)")

        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded {len(filtered_state_dict)} matching weight parameters")
    else:
        print("Pretrained weights file not found")


if __name__ == "__main__":
    import os
    # 测试模型创建
    model = create_swin_unetr_model(
        in_channels=1,
        out_channels=4,
        feature_size=48,
    )
    
    # 测试前向传播
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("模型创建成功！")

