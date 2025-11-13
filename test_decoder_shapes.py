import pytest
import torch

from models.ResNet18 import ResNet18Head, ResNet18Backbone, ResNet18Decoder
from models.ResNet50 import ResNet50Head, ResNet50Backbone, ResNet50Decoder
from models.VGG11 import VGG11Head, VGG11Backbone, VGG11Decoder
from models.VGG19 import VGG19Head, VGG19Backbone, VGG19Decoder
from models.DenseNet121 import DenseNet121Head, DenseNet121Backbone, DenseNet121Decoder

try:
    from models.ViT_B16 import ViTB16Head, ViTB16Backbone, ViTB16Decoder
    HAS_VIT = True
except Exception:  # pragma: no cover - timm may be missing
    HAS_VIT = False


MODEL_CONFIGS = {
    "resnet18": {
        "head": ResNet18Head,
        "backbone": ResNet18Backbone,
        "decoder": ResNet18Decoder,
        "cut_layers": [0, 1, 2, 3, 4],
    },
    "resnet50": {
        "head": ResNet50Head,
        "backbone": ResNet50Backbone,
        "decoder": ResNet50Decoder,
        "cut_layers": [0, 1, 2, 3, 4],
    },
    "vgg11": {
        "head": VGG11Head,
        "backbone": VGG11Backbone,
        "decoder": VGG11Decoder,
        "cut_layers": [0, 1, 2, 3, 4],
    },
    "vgg19": {
        "head": VGG19Head,
        "backbone": VGG19Backbone,
        "decoder": VGG19Decoder,
        "cut_layers": [0, 1, 2, 3],
    },
    "densenet121": {
        "head": DenseNet121Head,
        "backbone": DenseNet121Backbone,
        "decoder": DenseNet121Decoder,
        "cut_layers": [0, 1, 2, 3, 4],
    },
}

if HAS_VIT:
    MODEL_CONFIGS["vit_b16"] = {
        "head": ViTB16Head,
        "backbone": ViTB16Backbone,
        "decoder": ViTB16Decoder,
        "cut_layers": [0, 3, 7, 11],
    }


def _iter_cases():
    for name, cfg in MODEL_CONFIGS.items():
        for cut_layer in cfg["cut_layers"]:
            yield pytest.param(name, cut_layer, id=f"{name}-cut{cut_layer}")


@pytest.mark.parametrize("model_name,cut_layer", list(_iter_cases()))
def test_decoder_output_shape_matches_input(model_name, cut_layer):
    cfg = MODEL_CONFIGS[model_name]
    head_cls = cfg["head"]
    backbone_cls = cfg["backbone"]
    decoder_cls = cfg["decoder"]

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    try:
        head = head_cls(in_channels=3, cut_layer=cut_layer).eval()
        backbone = backbone_cls(cut_layer=cut_layer).eval()
        decoder = decoder_cls(cut_layer=cut_layer).eval()
    except RuntimeError as err:
        message = str(err).lower()
        if "download" in message or "timm" in message:
            pytest.skip(f"Skipping {model_name} cut_layer={cut_layer}: {err}")
        raise

    with torch.no_grad():
        smashed = head(dummy_input)
        backbone_output = backbone(smashed)
        reconstruction = decoder(backbone_output)

    assert reconstruction.shape == (batch_size, 3, 224, 224)


