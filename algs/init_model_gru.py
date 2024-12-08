import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from mora_params import *

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GRUAgent(nn.Module):
    def __init__(self, device):
        self.device = device
        super().__init__()
        self.gru = nn.GRU(64, 64)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.network = nn.Sequential(
            layer_init(nn.Linear(STATE_DIM, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.gru.input_size)),
            nn.ReLU(),
        )
            
        self.finger_actor = layer_init(nn.Linear(self.gru.hidden_size, len(POSSIBLE_FINGERS)), std=0.01)
        self.sum_actor = layer_init(nn.Linear(self.gru.hidden_size, len(POSSIBLE_AMOUNTS)), std=0.01)
        self.critic = layer_init(nn.Linear(self.gru.hidden_size, 1), std=1)

    def get_state(self, x, gru_state):
        hidden = self.network(x)
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        new_hidden, gru_state = self.gru(hidden, gru_state)
        return new_hidden, gru_state

    def get_states(self, sequences, gru_states):    
        padded_sequences = pad_sequence(sequences, batch_first=True).to(self.device)
        hidden = self.network(padded_sequences)
        sequence_lengths = [NUM_ROUNDS] * len(sequences)
        packed_sequences = pack_padded_sequence(hidden, sequence_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_sequences, gru_states)
        new_hidden, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        return new_hidden, output_lengths

    def get_action_and_value(self, sequence, gru_state, actions: set = None):
        if actions is None:
            hidden, gru_state = self.get_state(sequence, gru_state)
            finger_logits = self.finger_actor(hidden).squeeze(0)
            sum_logits = self.sum_actor(hidden).squeeze(0)

            finger_probs = Categorical(logits=finger_logits)
            sum_probs = Categorical(logits=sum_logits)

            finger_action = finger_probs.sample()
            sum_action = sum_probs.sample()

            return (finger_action, sum_action), \
                   (finger_probs.log_prob(finger_action), sum_probs.log_prob(sum_action)), \
                   (finger_probs.entropy(), sum_probs.entropy()), \
                   self.critic(hidden).view(gru_state.shape[1]), gru_state
        
        else:
            hidden, _ = self.get_states(sequence, gru_state)
            fingers_logits = self.finger_actor(hidden)
            sums_logits = self.sum_actor(hidden)
            values = self.critic(hidden)

            fingers_probs = Categorical(logits=fingers_logits)
            sums_probs = Categorical(logits=sums_logits)

            fingers_log_probs = fingers_probs.log_prob(actions[0])
            sums_log_probs = sums_probs.log_prob(actions[1])
            fingers_entropies = fingers_probs.entropy()
            sums_entropies = sums_probs.entropy()

            return None, \
                   (fingers_log_probs, sums_log_probs), \
                   (fingers_entropies, sums_entropies), \
                   values, \
                   None