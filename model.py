def vit_base_DINOv3_SEIA(
    img_size=(256, 128), IAS_dim=64, sse=True, **kwargs
):
    return SEIA(
        img_size=img_size,
        embed_dim=768,
        model_name="vit_base_patch16_dinov3.lvd1689m",
        IAS_dim=IAS_dim,
        sse=sse,
    )


def vit_large_DINOv3_SEIA(
    img_size=(256, 128), IAS_dim=64, sse=True, **kwargs
):
    return SEIA(
        img_size=img_size,
        embed_dim=1024,
        model_name="vit_large_patch16_dinov3.lvd1689m",
        IAS_dim=IAS_dim,
        sse=sse,
    )


