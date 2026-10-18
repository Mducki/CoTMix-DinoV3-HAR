import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3TimeSeriesEncoder(nn.Module):
    """
    Adapter: [B, 6, T] -> grayscale sensor-time map -> [B, 3, 224, 224]
    -> pretrained DINOv3 ViT-S/16 CLS embedding.

    The signal image conversion is a proposed adaptation for sensor data,
    not the official DINOv3 image preprocessing recipe.
    """
    def __init__(
        self,
        configs=None,
        repo_dir="dinov3",
        weights_path=None,
        pretrained=True,
        image_size=224,
        model_name="dinov3_vits16",
        freeze_backbone=False,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.repo_dir = str(repo_dir)
        self.model_name = model_name

        if pretrained and not weights_path:
            raise ValueError(
                "Pretrained DINOv3 weights are required. Pass weights_path to the .pth checkpoint."
            )

        self.backbone = torch.hub.load(
            self.repo_dir,
            self.model_name,
            source="local",
            weights=str(weights_path) if pretrained else None,
        )

        dim = getattr(self.backbone, "num_features", None)
        if dim is None:
            dim = getattr(self.backbone, "embed_dim", None)
        if dim is None:
            # Official DINOv3 ViT-S/16 has 384-dimensional embeddings.
            if self.model_name != "dinov3_vits16":
                raise RuntimeError(
                    "Cannot infer DINOv3 embedding dimension; set it explicitly for this model."
                )
            dim = 384
        self.output_dim = int(dim)

        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def signal_to_image(self, x):
        # x: [B, 6, T], six accelerometer/gyroscope channels.
        if x.ndim != 3:
            raise ValueError(f"Expected [B, 6, T], got {tuple(x.shape)}")
        if x.shape[1] != 6 and x.shape[2] == 6:
            x = x.transpose(1, 2)
        if x.shape[1] != 6:
            raise ValueError(f"Expected six sensor channels, got {tuple(x.shape)}")

        # Preserve sensor-axis ordering vertically and time horizontally.
        img = x.unsqueeze(1)  # [B, 1, 6, T]
        img = F.interpolate(
            img,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        # Map standardized signal to [0, 1] as grayscale and replicate to RGB.
        img = torch.sigmoid(img).repeat(1, 3, 1, 1)
        img = (img - self.rgb_mean) / self.rgb_std
        return img

    def forward(self, x):
        img = self.signal_to_image(x)
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(img)
            if isinstance(features, dict):
                if "x_norm_clstoken" in features:
                    return features["x_norm_clstoken"]
                if "x_norm_patchtokens" in features:
                    return features["x_norm_patchtokens"].mean(dim=1)
            if torch.is_tensor(features):
                if features.ndim == 3:
                    return features[:, 0]
                return features
        out = self.backbone(img)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            raise RuntimeError(f"Unexpected DINOv3 output keys: {list(out.keys())}")
        if out.ndim == 3:
            return out[:, 0]
        return out


class classifier(nn.Module):
    """Classifier compatible with CoTMix's feature_extractor -> classifier flow."""
    def __init__(self, configs):
        super().__init__()
        input_dim = int(getattr(configs, "features_dim", 384))
        num_classes = int(getattr(configs, "num_classes", 18))
        dropout = float(getattr(configs, "classifier_dropout", 0.1))
        self.logits = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.logits(x)


def get_backbone_class(backbone_name):
    if backbone_name in globals():
        return globals()[backbone_name]
    raise NotImplementedError(f"Backbone not found: {backbone_name}")
