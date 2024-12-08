import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from algs.init_model_gru import GRUAgent


class PPO_GRU:
    def __init__(self, 
                 learning_rate=0.001, 
                 batch_size=32,
                 train_epochs=10, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 actor_clip=0.1, 
                 critic_clip=0.1, 
                 ent_coef=0.0, 
                 alpha=0.1, 
                 device=None, 
                 log_dir=None):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.actor_clip = actor_clip
        self.critic_clip = critic_clip
        self.ent_coef = ent_coef
        self.alpha = alpha
        self.log_dir = log_dir

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if log_dir:
            self.log_dir = log_dir
            self.writer = SummaryWriter(self.log_dir)

        self.model = GRUAgent(device).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)

        self.train_steps = 0
        self.num_trains = 0
    
    def get_advantages_gae(self, rewards, values):
        advantages = []
        for ep_rewards, ep_values in zip(rewards, values):
            ep_len = len(ep_rewards)
            adv = torch.zeros(ep_len).to('cpu')
            last_gae_lambda = 0
            
            for t in reversed(range(ep_len)):
                if t == ep_len - 1:
                    next_value = 0
                else:
                    next_value = ep_values[t + 1]
                
                delta = ep_rewards[t] + self.gamma * next_value - ep_values[t]
                adv[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
                
            advantages.append(adv)
        
        return advantages
    
    def get_returns(self, advantages, values):
        return [adv + val for adv, val in zip(advantages, values)]

    def train(self, states, actions, rewards, logprobs, values):
        advantages = self.get_advantages_gae(rewards, values)
        y = self.get_returns(advantages, values)

        rollout_len = len(states)
        inds = np.arange(rollout_len)

        init_gru_states = torch.zeros(self.model.gru.num_layers, self.batch_size, self.model.gru.hidden_size).to(self.device)

        clipfracs = []
        for _ in range(self.train_epochs):
            np.random.shuffle(inds)
            for start in range(0, rollout_len, self.batch_size):
                end = start + self.batch_size
                b_inds = inds[start:end]

                b_states = [states[i] for i in b_inds]
                b_actions = [actions[i] for i in b_inds]
                b_finger_actions, b_sum_actions = (torch.tensor([[pair[i].item() for pair in row] for row in b_actions]).to(self.device) for i in (0, 1))

                _, b_newlogprobs, b_entropys, b_newvalues, _ = self.model.get_action_and_value(
                        b_states,
                        init_gru_states,
                        (b_finger_actions, b_sum_actions),
                )

                b_logprobs = [logprobs[i] for i in b_inds]
                b_finger_logprobs, b_sum_logprobs = (torch.tensor([[pair[i].item() for pair in row] for row in b_logprobs]).to(self.device) for i in (0, 1))

                finger_log_r = (b_newlogprobs[0] - b_finger_logprobs).flatten()
                sum_log_r = (b_newlogprobs[1] - b_sum_logprobs).flatten()
                
                finger_r = finger_log_r.exp()
                sum_r = sum_log_r.exp()

                with torch.no_grad():
                    finger_approx_kl = ((finger_r - 1) - finger_log_r).mean()
                    sum_approx_kl = ((sum_r - 1) - sum_log_r).mean()
                    clipfracs += [((finger_r - 1.0).abs() > self.actor_clip).float().mean().item(),
                                  ((sum_r - 1.0).abs() > self.actor_clip).float().mean().item()]

                b_advantages = torch.cat([advantages[i] for i in b_inds], dim=0).to(self.device)
                norm_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                finger_loss1 = -norm_advantages * finger_r
                finger_loss2 = -norm_advantages * torch.clamp(finger_r, 1 - self.actor_clip, 1 + self.actor_clip)
                finger_loss = torch.max(finger_loss1, finger_loss2).mean()

                sum_loss1 = -norm_advantages * sum_r
                sum_loss2 = -norm_advantages * torch.clamp(sum_r, 1 - self.actor_clip, 1 + self.actor_clip)
                sum_loss = torch.max(sum_loss1, sum_loss2).mean()

                p_loss = finger_loss + sum_loss

                b_values = torch.cat([values[i] for i in b_inds], dim=0).to(self.device)
                b_y = torch.cat([y[i] for i in b_inds], dim=0).to(self.device)

                v_loss_unclipped = (b_newvalues - b_y) ** 2
                v_clipped = b_values + torch.clamp(
                    b_newvalues - b_values,
                    -self.critic_clip,
                    self.critic_clip,
                )
                v_loss_clipped = (v_clipped - b_y) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = v_loss_max.mean()

                entropy_loss = (b_entropys[0].mean() + b_entropys[1].mean()) / 2
                loss = p_loss - self.ent_coef * entropy_loss + v_loss * self.alpha

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                if self.log_dir:
                    self.writer.add_scalar("losses/critic_loss", v_loss.detach().cpu().numpy(), self.train_steps)
                    self.writer.add_scalar("losses/actor_loss", p_loss.detach().cpu().numpy(), self.train_steps)
                    self.writer.add_scalar("losses/entropy_loss", entropy_loss.detach().cpu().numpy(), self.train_steps)
                    self.writer.add_scalar("losses/loss", loss.detach().cpu().numpy(), self.train_steps)
                    self.writer.add_scalar("losses/finger_approx_kl", finger_approx_kl.item(), self.train_steps)
                    self.writer.add_scalar("losses/sum_approx_kl", sum_approx_kl.item(), self.train_steps)
                    
                self.train_steps += 1

        if self.log_dir:
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.train_steps)

        self.num_trains += 1
