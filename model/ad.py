import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from env import map_dark_states, map_dark_states_inverse

class AD(torch.nn.Module):
    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']

        # TransformerEncoder (PyTorch) replacement for tiny_llama.Transformer
        # Use learned positional embeddings to mimic tiny-llama positional behavior.
        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_length, tf_n_embd))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_n_embd,
            nhead=tf_n_head,
            dim_feedforward=tf_dim_feedforward,
            activation='gelu',
            batch_first=True,  # so inputs are (batch, seq, emb)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tf_n_layer)

        # embeddings and heads (same as original)
        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

        # small init for pos_embedding
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _apply_positional_embedding(self, x):
        # x: (batch, seq, emb)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return x

    def transformer(self, x, max_seq_length=None, dtype=None):
        """
        Thin wrapper so existing code calling self.transformer(...) still works.
        We accept (batch, seq, emb) and return (batch, seq, emb).
        dtype argument is accepted for API compatibility - we won't dynamically change types here.
        """
        # Optionally cast for mixed precision; keep as float32 for stability of linear heads
        # If mixed_precision used elsewhere, automatic casting can handle it in training loop.
        x = self._apply_positional_embedding(x)
        out = self.transformer_encoder(x)  # (batch, seq, emb)
        return out

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

        # call our encoder wrapper
        transformer_output = self.transformer(context_embed,
                                              max_seq_length=self.max_seq_length,
                                              dtype=self.mixed_precision)

        result = {}

        logits_actions = self.pred_action(transformer_output[:, self.n_transit-1])  # (batch_size, dim_action)

        loss_full_action = self.loss_fn(logits_actions, target_actions)
        acc_full_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()

        result['loss_action'] = loss_full_action
        result['acc_action'] = acc_full_action

        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {}
        outputs['reward_episode'] = []

        reward_episode = np.zeros(vec_env.num_envs)

        # Get inital states embeddings
        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
        transformer_input = query_states_embed

        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach().to(torch.float)

            output = self.transformer(transformer_input,
                                        max_seq_length=self.max_seq_length,
                                        dtype='fp32')

            logits = self.pred_action(output[:, -1])

            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1)
                actions = rearrange(actions, 'e 1 -> e')
            else:
                actions = logits.argmax(dim=-1)

            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

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

        return outputs

    def beam_search(self, x, query_states, position, beam_k=5, sample=True):
        batch_size = x.size(0)

        output = self.transformer(x,
                                  max_seq_length=self.max_seq_length,
                                  dtype="fp32")

        logit_actions = self.pred_action(output[:, -1])

        if sample:
            log_probs = F.log_softmax(logit_actions, dim=-1)
            all_actions = torch.multinomial(log_probs.exp(), num_samples=self.config['num_actions'])
        else:
            all_actions = logit_actions.argsort(dim=-1, descending=True)  # (batch_size, num_actions)

        # Query all actions
        all_actions_embed = self.embed_query_action(all_actions)
        all_actions_embed = rearrange(all_actions_embed, 'b a h -> b a 1 h')

        x = repeat(x, 'b n h -> b a n h', a=self.config['num_actions'])
        x, _ = pack([x, all_actions_embed], 'b a * h')

        output = self.transformer(rearrange(x, 'b a n h -> (b a) n h'),
                                  max_seq_length=self.max_seq_length,
                                  dtype="fp32")

        output = rearrange(output, '(b a) n h -> b a n h', a=self.config['num_actions'])

        # Get rewards
        logits_rewards = self.pred_reward(output[:, :, -1])
        rewards = logits_rewards.argmax(dim=-1)  # (batch_size, num_actions)

        # Get next states
        logit_next_states = self.pred_next_state(output[:, :, -1])
        next_states = logit_next_states.argmax(dim=-1)  # (batch_size, num_actions)

        # Initialize cumulative rewards
        cum_rewards = rewards.clone().detach()

        # Sort actions according to rewards
        rewards_sort = cum_rewards.sort(dim=-1, descending=True, stable=True)
        cum_rewards = rewards_sort.values[:, :beam_k]
        indices_k = rewards_sort.indices[:, :beam_k]

        # Update cumulative rewards
        beam = torch.gather(all_actions, 1, indices_k)
        beam = rearrange(beam, 'b k -> b k 1')

        if self.config['env'] == 'darkroom':
            max_beam_steps = self.grid_size - 1
        elif self.config['env'] == 'darkkeytodoor' or self.config['env'] == 'darkroompermuted':
            max_beam_steps = (self.grid_size - 1) * 2
        else:
            raise ValueError('Invalid environment')

        position += 1
        beam_step = 1

        while position < self.config['horizon'] and beam_step < max_beam_steps:
            # Sort and cutoff variables
            x = torch.gather(x, 1, repeat(indices_k, 'b k -> b k n h', n=x.size(2), h=x.size(3)))
            actions_onehot = F.one_hot(beam[:, :, -1], num_classes=self.config['num_actions'])
            rewards = torch.gather(rewards, 1, indices_k)
            rewards = rearrange(rewards, 'b k -> b k 1')
            next_states = torch.gather(next_states, 1, indices_k)
            next_states_coord = map_dark_states_inverse(next_states, self.config['grid_size'])
            query_states = repeat(query_states, 'b k d -> b (k a) d', a=self.config['num_actions'])
            query_states = torch.gather(query_states, 1, repeat(indices_k, 'b k -> b k d', d=query_states.size(2)))

            # Make new context transition
            new_context, _ = pack([query_states, actions_onehot, rewards, next_states_coord], 'b k *')
            new_context_embed = self.embed_context(new_context.float())
            new_context_embed = repeat(new_context_embed, 'b k h -> b (k a) 1 h', a=self.config['num_actions'])

            # Make new query states
            query_states_embed = self.embed_query_state(next_states)
            query_states_embed = repeat(query_states_embed, 'b k h -> b (k a) 1 h', a=self.config['num_actions'])

            query_states = next_states_coord  # (batch_size, beam_k, dim_state)

            # Make transformer input
            x = repeat(x, 'b k n h -> b (k a) n h', a=self.config['num_actions'])

            all_actions = torch.arange(self.config['num_actions'], device=self.device)
            all_actions_embed = self.embed_query_action(all_actions)
            all_actions_embed = repeat(all_actions_embed, 'a h -> b (k a) 1 h', b=batch_size, k=rewards.size(1))

            x, _ = pack([x[:, :, 1:self.config['n_transit']-1], new_context_embed, query_states_embed, all_actions_embed], 'b ka * h')

            assert x.size(2) == self.config['n_transit'] + 1

            # query states & actions
            output = self.transformer(rearrange(x, 'b ka n h -> (b ka) n h'),
                                      max_seq_length=self.max_seq_length,
                                      dtype="fp32")

            output = rearrange(output, '(b ka) n h -> b ka n h', b=batch_size)

            # Get rewards
            logit_rewards = self.pred_reward(output[:, :, -1])
            rewards = logit_rewards.argmax(dim=-1)  # (batch_size, beam_k * num_actions)

            # Get next states
            logit_next_states = self.pred_next_state(output[:, :, -1])
            next_states = logit_next_states.argmax(dim=-1)  # (batch_size, beam_k * num_actions)

            # Update cumulative rewards
            cum_rewards = repeat(cum_rewards, 'b k -> b (k a)', a=self.config['num_actions'])
            cum_rewards = cum_rewards + rewards
            rewards_sort = cum_rewards.sort(dim=-1, descending=True, stable=True)
            cum_rewards = rewards_sort.values[:, :beam_k]
            indices_k = rewards_sort.indices[:, :beam_k]

            new_actions = repeat(all_actions, 'a -> b (k a) 1', b=batch_size, k=beam.size(1))
            beam = repeat(beam, 'b k s -> b (k a) s', a=self.config['num_actions'])
            beam, _ = pack([beam, new_actions], 'b ka *')
            beam = torch.gather(beam, 1, repeat(indices_k, 'b k -> b k s', s=beam.size(2)))

            position += 1
            beam_step += 1

        return beam[:, 0, 0]
