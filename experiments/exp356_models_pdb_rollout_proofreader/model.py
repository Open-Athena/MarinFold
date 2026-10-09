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
BIDIRECTIONAL_ARCHITECTURE = 'exp277-bidirectional-contact-recall-v1'
FROZEN_ARCHITECTURE = 'exp277-frozen-causal-contact-encoder-v2'


class Proofreader(nn.Module):
    """Encode all supplied evidence and classify each proposed contact."""

    architecture = BIDIRECTIONAL_ARCHITECTURE

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
        (path / 'proofreader.json').write_text(json.dumps(dict(metadata,architecture=self.architecture), indent=2))


class ContactContext(nn.Module):
    """Mix frozen contact-triple features across the complete supplied prefix."""

    def __init__(self, backbone_width: int, width: int, layers: int, heads: int, max_positions: int):
        super().__init__()
        self.triple_projection = nn.Linear(3*backbone_width,width)
        self.global_projection = nn.Linear(2*backbone_width+2,width)
        self.order_embedding = nn.Embedding(max_positions,width)
        nn.init.normal_(self.order_embedding.weight,std=.02)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(width,heads,4*width,dropout=0.,
            activation='gelu',batch_first=True,norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Sequential(nn.Linear(width,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self, triples: torch.Tensor, global_features: torch.Tensor,
                contact_mask: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """Return per-contact logits and a pooled bidirectional representation."""
        contacts = self.triple_projection(triples)
        order = torch.arange(contacts.shape[1],device=contacts.device)
        contacts = contacts+self.order_embedding(order)[None]
        pooled = self.global_projection(global_features)[:,None]
        values = torch.cat([pooled,contacts],dim=1)
        mask = torch.cat([torch.ones_like(contact_mask[:,:1]),contact_mask],dim=1)
        for layer in self.encoder:
            values = layer(values,src_key_padding_mask=~mask,is_causal=False)
        encoded = self.norm(values)
        return self.classifier(encoded[:,1:]).squeeze(-1).float(),encoded[:,0]


class CausalContactProofreader(Proofreader):
    """Preserve generator features while learning bidirectional contact reasoning.

    The frozen causal backbone processes the actual sequence and emitted tokens.
    A separate encoder sees all contact triples in the supplied prefix, so later
    contacts can revise earlier readouts without changing the generator features.
    """

    architecture = FROZEN_ARCHITECTURE

    def __init__(self, backbone: Qwen3Model, begin_statement_id: int, readout_width: int = 512,
                 readout_layers: int = 4, readout_heads: int = 8):
        super().__init__(backbone)
        self.readout_config = dict(begin_statement_id=begin_statement_id,readout_width=readout_width,
            readout_layers=readout_layers,readout_heads=readout_heads)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.contact_head = ContactContext(backbone.config.hidden_size,readout_width,readout_layers,
            readout_heads,backbone.config.max_position_embeddings)
        self.recall_head = nn.Sequential(nn.Linear(readout_width,128),nn.SiLU(),nn.Linear(128,1))

    def train(self, mode: bool = True):
        """Keep the pretrained feature extractor deterministic and frozen."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids: torch.Tensor, token_mask: torch.Tensor,
                contact_positions: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """Read causal triple features, then allow full contact-to-contact attention."""
        with torch.no_grad():
            hidden = self.backbone(input_ids=input_ids,attention_mask=token_mask,use_cache=False).last_hidden_state
        batch = torch.arange(input_ids.shape[0],device=input_ids.device)
        contact_mask = contact_positions>0  # Collation reserves zero for padded readouts.
        indices = torch.stack([contact_positions-2,contact_positions-1,contact_positions],dim=-1).clamp_min(0)
        triples = hidden[batch[:,None,None],indices].flatten(-2)
        prompt_end = (input_ids==self.readout_config['begin_statement_id']).long().argmax(dim=1)
        prefix_end = token_mask.sum(dim=1)-2  # Exclude the appended assessment token.
        counts = torch.stack([prompt_end+1,contact_mask.sum(dim=1)],dim=-1).float().log1p()
        global_features = torch.cat([hidden[batch,prompt_end],hidden[batch,prefix_end],counts],dim=-1)
        logits,pooled = self.contact_head(triples,global_features,contact_mask)
        return logits,self.recall_head(pooled).squeeze(-1).float().sigmoid()

    def save(self, path: Path, tokenizer, metadata: dict) -> None:
        """Save the frozen backbone and all trainable contact-encoder parameters."""
        super().save(path,tokenizer,dict(metadata,readout_config=self.readout_config))


def load_model(path: Path, *, initialize: bool = False, architecture: str | None = None) -> tuple[Proofreader, object]:
    """Load either the frozen generator initialization or a saved proofreader."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    metadata = {} if initialize else json.loads((path/'proofreader.json').read_text())
    actual_architecture = (architecture or BIDIRECTIONAL_ARCHITECTURE) if initialize else metadata['architecture']
    if architecture is not None and actual_architecture!=architecture:
        raise ValueError('Requested architecture differs from the checkpoint')
    config = Qwen3Config.from_pretrained(path)
    if set(config.layer_types) != {'full_attention'}:
        raise ValueError('Only full-attention Qwen3 backbones are supported')
    backbone = Qwen3Model.from_pretrained(path, config=config, dtype=torch.float32, attn_implementation='sdpa')
    if initialize:
        tokenizer.add_special_tokens({'additional_special_tokens': [ASSESSMENT]})
        backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    elif ASSESSMENT not in tokenizer.get_vocab():
        raise ValueError('Checkpoint tokenizer lacks the assessment token')
    if actual_architecture == BIDIRECTIONAL_ARCHITECTURE:
        model = Proofreader(backbone)
    elif actual_architecture == FROZEN_ARCHITECTURE:
        readout_config = dict(begin_statement_id=tokenizer.convert_tokens_to_ids('<begin_statements>')) if initialize else metadata['readout_config']
        model = CausalContactProofreader(backbone,**readout_config)
    else:
        raise ValueError(f'Unknown proofreading architecture: {actual_architecture}')
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
