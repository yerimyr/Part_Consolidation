from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


class EdgeAwareMessagePassing(nn.Module):
    def __init__(self, emb_dim: int, edge_feat_dim: int):
        super().__init__()
        self.node_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.edge_proj = nn.Linear(edge_feat_dim, emb_dim, bias=False)
        self.weight_proj = nn.Linear(1, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, h: torch.Tensor, edge_features: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        hi = self.node_proj(h).unsqueeze(2)      # [B, N, 1, E]
        ej = self.edge_proj(edge_features)       # [B, N, N, E]
        wij = self.weight_proj(W.unsqueeze(-1))  # [B, N, N, E]

        msg = torch.tanh(hi + ej + wij)
        agg = msg.mean(dim=2)
        return self.out_proj(agg)


class PCPolicy(nn.Module):
    """
    Policy for merge-based part consolidation.

    Output:
    - action 0: STOP
    - action 1..K: merge one fixed group-slot pair
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        emb_dim: int = 256,
        num_message_passing: int = 3,
        temperature: float = 1.2,
        num_decoder_layers: int = 2,
        num_decoder_heads: int = 4,
    ):
        super().__init__()

        self.base_node_feat_dim = node_feat_dim
        # valid_part, normalized current group cardinality
        self.dynamic_node_feat_dim = 2
        self.total_node_feat_dim = self.base_node_feat_dim + self.dynamic_node_feat_dim
        self.temperature = float(temperature)

        self.node_embed = nn.Linear(self.total_node_feat_dim, emb_dim)
        self.layers = nn.ModuleList(
            [EdgeAwareMessagePassing(emb_dim, edge_feat_dim) for _ in range(num_message_passing)]
        )

        self.pair_scorer = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, 1),
        )
        self.stop_scorer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, 1),
        )
        self.logit_bias = nn.Parameter(torch.zeros(1))
        self._pair_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _pair_tensors(self, max_parts: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (max_parts, device)
        if key not in self._pair_cache:
            pairs = [(i, j) for i in range(max_parts) for j in range(i + 1, max_parts)]
            self._pair_cache[key] = (
                torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device),
                torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device),
            )
        return self._pair_cache[key]

    def build_dynamic_node_features(self, td: TensorDict) -> torch.Tensor:
        B, N, _ = td["node_features"].shape

        group_id = td["group_id"]
        valid_part = td.get("valid_part_mask", group_id.ge(0)).float().unsqueeze(-1)
        max_parts = N - 1
        valid_group = group_id.ge(0)
        group_idx = group_id.clamp(min=0, max=max_parts - 1)
        group_counts = torch.zeros((B, max_parts), dtype=torch.float32, device=group_id.device)
        group_counts.scatter_add_(1, group_idx, valid_group.float())
        group_card = group_counts.gather(1, group_idx) * valid_group.float()
        group_card_ratio = (group_card / max(float(N - 1), 1.0)).unsqueeze(-1)

        dyn = torch.cat([valid_part, group_card_ratio], dim=-1)
        return dyn

    def encode(self, td: TensorDict) -> torch.Tensor:
        x_static = td["node_features"].float()
        x_dynamic = self.build_dynamic_node_features(td)
        x = torch.cat([x_static, x_dynamic], dim=-1)

        e = td["edge_features"].float()
        W = td["W"].float()

        h = self.node_embed(x)
        for layer in self.layers:
            h = h + layer(h, e, W)
        return h

    def compute_logits(self, node_emb: torch.Tensor, td: TensorDict) -> torch.Tensor:
        B, N, E = node_emb.shape
        group_id = td["group_id"]
        action_mask = td["action_mask"]
        num_actions = action_mask.size(-1)
        max_parts = N - 1

        valid = td.get("valid_part_mask", group_id.ge(0))
        part_mask = valid[:, 1:].float()
        global_denom = part_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        global_context = (node_emb[:, 1:, :] * part_mask.unsqueeze(-1)).sum(dim=1) / global_denom

        logits = torch.full((B, num_actions), -1e9, dtype=node_emb.dtype, device=node_emb.device)
        logits[:, 0] = self.stop_scorer(global_context).squeeze(-1)

        if num_actions > 1:
            valid_group = group_id.ge(0) & valid.bool()
            group_idx = group_id.clamp(min=0, max=max_parts - 1)
            membership = F.one_hot(group_idx, num_classes=max_parts).to(node_emb.dtype)
            membership = membership * valid_group.unsqueeze(-1).to(node_emb.dtype)
            group_counts = membership.sum(dim=1).clamp_min(1.0)
            group_emb = torch.einsum("bng,bne->bge", membership, node_emb) / group_counts.unsqueeze(-1)

            pair_ga, pair_gb = self._pair_tensors(max_parts, node_emb.device)
            num_pair_actions = min(num_actions - 1, pair_ga.numel())
            pair_ga = pair_ga[:num_pair_actions]
            pair_gb = pair_gb[:num_pair_actions]
            ha = group_emb.index_select(1, pair_ga)
            hb = group_emb.index_select(1, pair_gb)
            feat = torch.cat([ha, hb, torch.abs(ha - hb), ha * hb], dim=-1)
            logits[:, 1 : 1 + num_pair_actions] = self.pair_scorer(feat).squeeze(-1)

        logits = logits + self.logit_bias
        return logits

    def act(self, td: TensorDict, sample: bool = True, epsilon: float = 0.0):
        node_emb = self.encode(td)
        logits = self.compute_logits(node_emb, td)
        logits = logits / self.temperature

        mask = td["action_mask"].clone()
        no_valid = ~mask.any(dim=-1)
        if no_valid.any():
            mask[no_valid, 0] = True
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits, dim=-1)
        B = probs.size(0)

        if sample:
            if epsilon > 0.0:
                random_pick = torch.rand(B, device=probs.device) < epsilon
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)
                if random_pick.any():
                    rows = torch.arange(B, device=probs.device)[random_pick]
                    valid = mask[rows].float()
                    valid = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
                    rand_action = torch.multinomial(valid, num_samples=1).squeeze(-1)
                    action[rows] = rand_action
            else:
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            action = torch.argmax(probs, dim=-1)

        chosen_prob = probs.gather(1, action.view(-1, 1)).clamp_min(1e-12)
        logp = torch.log(chosen_prob).squeeze(-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        return action, logp, entropy, logits

    def forward(self, td: TensorDict, max_steps: int = 1, sample: bool = True, epsilon: float = 0.0):
        actions, logps, entropies = [], [], []
        for _ in range(max_steps):
            action, logp, entropy, _ = self.act(td, sample=sample, epsilon=epsilon)
            actions.append(action)
            logps.append(logp)
            entropies.append(entropy)
        return (
            torch.stack(actions, dim=1),
            torch.stack(logps, dim=1),
            torch.stack(entropies, dim=1),
        )


def make_pc_policy(**kwargs):
    return PCPolicy(
        node_feat_dim=kwargs.get("node_feat_dim", 10),
        edge_feat_dim=kwargs.get("edge_feat_dim", 7),
        emb_dim=kwargs.get("emb_dim", 128),
        num_message_passing=kwargs.get("num_message_passing", 3),
        temperature=kwargs.get("temperature", 1.2),
        num_decoder_layers=kwargs.get("num_decoder_layers", 2),
        num_decoder_heads=kwargs.get("num_decoder_heads", 4),
    )
