import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from models.models import DINOv3TimeSeriesEncoder, classifier


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(f"Algorithm not found: {algorithm_name}")
    return globals()[algorithm_name]


class Algorithm(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class CoTMix(Algorithm):
    """
    CoTMix-style UDA update adapted to DINOv3 features.
    Target labels are intentionally not accepted by update().
    Core losses follow the public CoTMix implementation:
    source CE, target conditional entropy, source supervised contrastive,
    target cross-view contrastive, with temporal mixup.
    """
    def __init__(
        self,
        backbone_fe=None,
        configs=None,
        hparams=None,
        device="cuda",
        dinov3_repo="dinov3",
        dinov3_weights=None,
        pretrained=True,
        image_size=224,
        num_classes=18,
    ):
        super().__init__(configs)
        hparams = hparams or {}
        self.hparams = hparams
        self.feature_extractor = DINOv3TimeSeriesEncoder(
            configs=configs,
            repo_dir=dinov3_repo,
            weights_path=dinov3_weights,
            pretrained=pretrained,
            image_size=image_size,
        )
        dim = self.feature_extractor.output_dim
        cfg = SimpleNamespace(features_dim=dim, num_classes=num_classes, classifier_dropout=0.1)
        self.classifier = classifier(cfg)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.AdamW(
            (p for p in self.network.parameters() if p.requires_grad),
            lr=float(hparams.get("learning_rate", 1e-5)),
            weight_decay=float(hparams.get("weight_decay", 1e-4)),
        )
        self.device_hint = device

    @staticmethod
    def conditional_entropy(logits):
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        return -(probs * log_probs).sum(dim=1).mean()

    @staticmethod
    def nt_xent(z1, z2, temperature=0.2):
        # Symmetric InfoNCE, one positive paired view per sample.
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        n = z1.shape[0]
        if n < 2:
            return F.mse_loss(z1, z2)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / temperature
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        targets = torch.cat(
            [torch.arange(n, 2 * n, device=z.device),
             torch.arange(0, n, device=z.device)]
        )
        return F.cross_entropy(sim, targets)

    @staticmethod
    def supervised_contrastive(z_views, labels, temperature=0.2):
        # z_views [B, 2, D]; same-class samples are positives.
        b, v, d = z_views.shape
        z = F.normalize(z_views.reshape(b * v, d), dim=1)
        labels = labels.view(-1, 1)
        labels = labels.repeat_interleave(v, dim=0)
        logits = torch.matmul(z, z.T) / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(b * v, device=z.device, dtype=torch.bool)
        positives = labels.eq(labels.T) & ~self_mask
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positives.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return z.sum() * 0.0
        mean_log_prob_pos = (log_prob * positives).sum(dim=1) / positive_count.clamp_min(1)
        return -mean_log_prob_pos[valid].mean()

    def temporal_mixup(self, src_x, trg_x):
        mix_ratio = float(self.hparams.get("mix_ratio", 0.5))
        temporal_shift = int(self.hparams.get("temporal_shift", 10))
        h = max(1, temporal_shift // 2)
        src_roll = torch.stack(
            [torch.roll(trg_x, shifts=-i, dims=2) for i in range(-h, h)], dim=2
        ).mean(dim=2)
        trg_roll = torch.stack(
            [torch.roll(src_x, shifts=-i, dims=2) for i in range(-h, h)], dim=2
        ).mean(dim=2)
        src_dominant = mix_ratio * src_x + (1.0 - mix_ratio) * src_roll
        trg_dominant = mix_ratio * trg_x + (1.0 - mix_ratio) * trg_roll
        return src_dominant, trg_dominant

    def update(self, src_x, src_y, trg_x):
        # IMPORTANT: no target labels are used in this method.
        self.train()
        if src_x.shape[0] != trg_x.shape[0]:
            n = min(src_x.shape[0], trg_x.shape[0])
            src_x, src_y, trg_x = src_x[:n], src_y[:n], trg_x[:n]

        src_dominant, trg_dominant = self.temporal_mixup(src_x, trg_x)
        self.optimizer.zero_grad(set_to_none=True)

        src_feat = self.feature_extractor(src_x)
        src_logits = self.classifier(src_feat)
        trg_feat = self.feature_extractor(trg_x)
        trg_logits = self.classifier(trg_feat)
        src_dom_feat = self.feature_extractor(src_dominant)
        src_dom_logits = self.classifier(src_dom_feat)
        trg_dom_feat = self.feature_extractor(trg_dominant)
        trg_dom_logits = self.classifier(trg_dom_feat)

        src_cls = F.cross_entropy(src_logits, src_y)
        trg_ent = self.conditional_entropy(trg_logits)
        src_supcon = self.supervised_contrastive(
            torch.stack([src_logits, src_dom_logits], dim=1), src_y
        )
        trg_con = self.nt_xent(trg_logits, trg_dom_logits, temperature=0.2)

        loss = (
            src_cls * float(self.hparams.get("src_cls_weight", 1.0))
            + trg_ent * float(self.hparams.get("trg_entropy_weight", 0.1))
            + src_supcon * float(self.hparams.get("src_supCon_weight", 0.1))
            + trg_con * float(self.hparams.get("trg_cont_weight", 0.1))
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()

        return {
            "Total_loss": float(loss.detach().cpu()),
            "src_cls_loss": float(src_cls.detach().cpu()),
            "trg_entropy_loss": float(trg_ent.detach().cpu()),
            "src_supcon_loss": float(src_supcon.detach().cpu()),
            "trg_con_loss": float(trg_con.detach().cpu()),
        }
