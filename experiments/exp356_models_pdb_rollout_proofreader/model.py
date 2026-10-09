"""Bidirectional exp277 encoder with parallel contact and recall readouts.

Qwen3Model's public precomputed-mask mapping bypasses causal mask creation.
Passing is_causal=False also disables SDPA's implicit causal flag when an
unpadded batch needs no explicit mask. No imported modules are monkey-patched.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

ASSESSMENT = '<assess>'


class Proofreader(nn.Module):
    """Encode all supplied evidence and classify each proposed contact."""

    def __init__(self, backbone: Qwen3Model):
        super().__init__()
        self.backbone = backbone
        width = backbone.config.hidden_size
        self.contact_head = nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, 1))
        self.recall_head = nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, input_ids: torch.Tensor, token_mask: torch.Tensor,
                contact_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contact logits and predicted current recall, with full attention."""
        # A broadcast key mask is sufficient: padded query outputs are never read.
        # The unpadded path permits PyTorch's flash SDPA kernel and grouped heads.
        mask = None if bool(token_mask.all()) else token_mask[:, None, None, :]
        hidden = self.backbone(input_ids=input_ids, attention_mask={'full_attention': mask},
                               is_causal=False, use_cache=False).last_hidden_state
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        contacts = hidden[batch[:, None], contact_positions]
        assessment = hidden[batch, token_mask.sum(dim=1)-1]
        return self.contact_head(contacts).squeeze(-1).float(), self.recall_head(assessment).squeeze(-1).float().sigmoid()

    def save(self, path: Path, tokenizer, metadata: dict) -> None:
        """Save a self-contained encoder, heads, tokenizer and provenance."""
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path)
        tokenizer.save_pretrained(path)
        torch.save({'contact_head': self.contact_head.state_dict(),
                    'recall_head': self.recall_head.state_dict()}, path / 'heads.pt')
        (path / 'proofreader.json').write_text(json.dumps(metadata, indent=2))


def load_model(path: Path, *, initialize: bool = False) -> tuple[Proofreader, object]:
    """Load either the frozen generator initialization or a saved proofreader."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    config = Qwen3Config.from_pretrained(path)
    if set(config.layer_types) != {'full_attention'}:
        raise ValueError('Only full-attention Qwen3 backbones are supported')
    backbone = Qwen3Model.from_pretrained(path, config=config, dtype=torch.float32, attn_implementation='sdpa')
    if initialize:
        tokenizer.add_special_tokens({'additional_special_tokens': [ASSESSMENT]})
        backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    elif ASSESSMENT not in tokenizer.get_vocab():
        raise ValueError('Checkpoint tokenizer lacks the assessment token')
    model = Proofreader(backbone)
    if not initialize:
        heads = torch.load(path / 'heads.pt', map_location='cpu', weights_only=True)
        model.contact_head.load_state_dict(heads['contact_head'])
        model.recall_head.load_state_dict(heads['recall_head'])
    return model, tokenizer


def loss_and_metrics(logits: torch.Tensor, recall: torch.Tensor, batch: dict,
                     recall_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average contact loss per example, then add supervised recall regression."""
    mask = batch['contact_mask'].float()
    unique = batch['unique_mask'].float() * mask
    bce = F.binary_cross_entropy_with_logits(logits, batch['labels'], reduction='none')
    contact_loss = ((bce * mask).sum(-1) / mask.sum(-1)).mean()
    recall_loss = F.mse_loss(recall, batch['recall'])
    probabilities = logits.sigmoid()
    precision = (probabilities * unique).sum(-1) / unique.sum(-1)
    metrics = dict(contact_loss=contact_loss.detach(), recall_mse=recall_loss.detach(),
        precision_mae=(precision-batch['precision']).abs().mean().detach(),
        recall_mae=(recall-batch['recall']).abs().mean().detach(),
        brier=(((probabilities-batch['labels']).square()*mask).sum(-1)/mask.sum(-1)).mean().detach())
    return contact_loss + recall_weight * recall_loss, metrics
