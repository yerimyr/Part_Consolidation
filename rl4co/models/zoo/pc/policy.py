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
        emb_dim: int = 128,
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

    def build_dynamic_node_features(self, td: TensorDict) -> torch.Tensor:
        B, N, _ = td["node_features"].shape

        group_id = td["group_id"]
        valid_part = td.get("valid_part_mask", group_id.ge(0)).float().unsqueeze(-1)
        group_card = torch.zeros((B, N), dtype=torch.float32, device=group_id.device)
        for b in range(B):
            ids = torch.unique(group_id[b][group_id[b] >= 0])
            for gid in ids.tolist():
                members = group_id[b].eq(int(gid))
                group_card[b, members] = members.float().sum()
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
        pair_list = [(i, j) for i in range(max_parts) for j in range(i + 1, max_parts)]

        valid = td.get("valid_part_mask", group_id.ge(0))
        part_mask = valid[:, 1:].float()
        global_denom = part_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        global_context = (node_emb[:, 1:, :] * part_mask.unsqueeze(-1)).sum(dim=1) / global_denom

        logits = torch.full((B, num_actions), -1e9, dtype=node_emb.dtype, device=node_emb.device)
        logits[:, 0] = self.stop_scorer(global_context).squeeze(-1)

        for b in range(B):
            group_emb: dict[int, torch.Tensor] = {}
            ids = sorted({int(g) for g in group_id[b, 1:][valid[b, 1:]].tolist() if int(g) >= 0})
            for gid in ids:
                members = (group_id[b] == gid) & valid[b]
                denom = members.float().sum().clamp_min(1.0)
                group_emb[gid] = (node_emb[b] * members.float().unsqueeze(-1)).sum(dim=0) / denom

            for action_idx, (ga, gb) in enumerate(pair_list, start=1):
                if action_idx >= num_actions:
                    break
                if ga not in group_emb or gb not in group_emb:
                    continue
                ha = group_emb[ga]
                hb = group_emb[gb]
                feat = torch.cat([ha, hb, torch.abs(ha - hb), ha * hb], dim=-1)
                logits[b, action_idx] = self.pair_scorer(feat).squeeze(-1)

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
