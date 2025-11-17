import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from typing import List, Tuple

from env import map_dark_states, map_dark_states_inverse

class AD(torch.nn.Module):

    class TransformerEncoderLayerWithAttn(nn.Module):
        """
        A thin reimplementation of TransformerEncoderLayer forward that
        returns both the output and the attention weights from self-attention.
        This mirrors PyTorch's layer behaviour (norms, dropouts, feedforward),
        but explicitly asks self_attn for weights (need_weights=True).
        """
        def __init__(self, base_layer: nn.TransformerEncoderLayer):
            super().__init__()
            # copy modules from provided base_layer instance
            # base_layer is an instance of nn.TransformerEncoderLayer you created
            self.self_attn = base_layer.self_attn
            self.linear1 = base_layer.linear1
            self.dropout = base_layer.dropout
            self.linear2 = base_layer.linear2
            self.norm1 = base_layer.norm1
            self.norm2 = base_layer.norm2
            self.dropout1 = base_layer.dropout1
            self.dropout2 = base_layer.dropout2
            self.activation = base_layer.activation

        def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns:
                src_out: Tensor shape (B, L, D)
                attn_weights: Tensor shape (B, num_heads, L, L)
            """
            # Self-attention: ask for weights directly
            # note: average_attn_weights=False to get per-head weights (PyTorch >=1.11)
            attn_output, attn_weights = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False
            )  # attn_output: (B, L, D), attn_weights: (B, num_heads, L, L) or (B, L, L) depending on PyTorch version
            # ensure attn_weights has shape (B, heads, L, L)
            if attn_weights.dim() == 3:
                # older behaviour: (B, L, L) => no per-head; treat as single head
                attn_weights = attn_weights.unsqueeze(1)

            src = src + self.dropout1(attn_output)
            src = self.norm1(src)
            # feedforward
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src, attn_weights


    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']

        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_length, tf_n_embd))

        base_layer = nn.TransformerEncoderLayer(
            d_model=tf_n_embd,
            nhead=tf_n_head,
            dim_feedforward=tf_dim_feedforward,
            activation='gelu',
            batch_first=True,
        )

        self.encoder_layers = nn.ModuleList([
            AD.TransformerEncoderLayerWithAttn(base_layer) for _ in range(tf_n_layer)
        ])

        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _apply_positional_embedding(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return x

    def transformer(self, x, max_seq_length=None, dtype=None):
        """
        Returns:
            out: (batch, seq, emb)
            attentions: list length=num_layers; each tensor (B, H, L, L)
        """
        x = self._apply_positional_embedding(x)
        attentions = []
        src = x
        # optional masks: you can pass src_mask or src_key_padding_mask if needed
        for layer in self.encoder_layers:
            src, attn = layer(src)
            # attn shape -> (B, heads, L, L)
            attentions.append(attn)
        return src, attentions

    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_state)
        target_actions = x['target_actions'].to(self.device)  # (batch_size,)
        states = x['states'].to(self.device)  # (batch_size, num_transit, dim_state)
        actions = x['actions'].to(self.device)  # (batch_size, num_transit, num_actions)
        next_states = x['next_states'].to(self.device)  # (batch_size, num_transit, dim_state)
        rewards = x['rewards'].to(self.device)  # (batch_size, num_transit)
        rewards = rearrange(rewards, 'b n -> b n 1')

        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size).to(torch.long))
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')

        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)
        context_embed, _ = pack([context_embed, query_states_embed], 'b * d')

        transformer_output, attentions = self.transformer(context_embed,
                                              max_seq_length=self.max_seq_length,
                                              dtype=self.mixed_precision)

        result = {}

        logits_actions = self.pred_action(transformer_output[:, self.n_transit-1])  # (batch_size, dim_action)

        loss_full_action = self.loss_fn(logits_actions, target_actions)
        acc_full_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()

        result['loss_action'] = loss_full_action
        result['acc_action'] = acc_full_action
        result['attentions'] = attentions

        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True, return_attentions=False):
        outputs = {}
        outputs['reward_episode'] = []

        reward_episode = np.zeros(vec_env.num_envs)

        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
        transformer_input = query_states_embed

        if return_attentions:
            per_step_attentions = []
            dones_history = []

        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach().to(torch.float)

            output, attentions = self.transformer(transformer_input,
                                        max_seq_length=self.max_seq_length,
                                        dtype='fp32')

            if return_attentions:
                per_step_attentions.append([a.detach().cpu().clone() for a in attentions])

            logits = self.pred_action(output[:, -1])

            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1)
                actions = rearrange(actions, 'e 1 -> e')
            else:
                actions = logits.argmax(dim=-1)

            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            if return_attentions:
                dones_history.append(dones.copy())

            actions = rearrange(actions, 'e -> e 1 1')
            actions = F.one_hot(actions, num_classes=self.config['num_actions'])

            reward_episode += rewards
            rewards = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards = rearrange(rewards, 'e -> e 1 1')

            query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
            query_states = rearrange(query_states, 'e d -> e 1 d')

            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)

                states_next = torch.tensor(np.stack([info['terminal_observation'] for info in infos]),
                                           device=self.device, dtype=torch.float)

                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach().to(torch.float)

            query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))

            context, _ = pack([query_states_prev, actions, rewards, states_next], 'e i *')
            context_embed = self.embed_context(context)

            if transformer_input.size(1) > 1:
                context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
                context_embed = context_embed[:, -(self.n_transit-1):]

            transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)

        if return_attentions:
            outputs['attentions'] = per_step_attentions
            outputs['dones_history'] = dones_history

        return outputs
