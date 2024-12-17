from mora_params import *
import random
import torch
from algs.init_model_gru import GRUAgent

class MoraPlayer:
    def __init__(self, 
                 name: str, 
                 model: GRUAgent):
        self.name = name
        self.model = model
        self.replay_buffer = []

        self.game_states = []
        self.game_actions = []
        self.game_logprobs = []
        self.game_values = []
  
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.action_masks = []

        self._device = 'cpu'
        self.next_gru_state = torch.zeros(self.model.gru.num_layers, 1, self.model.gru.hidden_size).to(self._device)

    def _init_state(self, data):
        if data:
            last_round_data = data[-1].copy()
            last_round_data.pop(self.name)
            return [last_round_data[player]['finger'] / max(POSSIBLE_FINGERS) for player in last_round_data.keys()] + \
                   [last_round_data[player]['amount'] / max(POSSIBLE_AMOUNTS) for player in last_round_data.keys()]
        
        return [-1] * STATE_DIM
        
    def get_action(self, data):
        next_state = torch.Tensor(self._init_state(data)).to(self._device)
        with torch.no_grad():
            actions, logprobs, _, value, self.next_gru_state = self.model.get_action_and_value(
                next_state,
                self.next_gru_state
            )

        self.game_states.append(next_state)
        self.game_actions.append(actions)
        self.game_logprobs.append(logprobs)
        self.game_values.append(value)

        return {'finger': POSSIBLE_FINGERS[actions[0]], 'amount': POSSIBLE_AMOUNTS[actions[1]]}

    def set_buffer(self, game_state):
        # self.replay_buffer.append(game_state)
        self.next_gru_state = torch.zeros(self.model.gru.num_layers, 1, self.model.gru.hidden_size).to('cpu')
        game_rewards = [round_dict[self.name]['score'] for round_dict in game_state]

        self.states.append(torch.stack(self.game_states))
        self.actions.append(self.game_actions)
        self.rewards.append(torch.Tensor(game_rewards))
        self.logprobs.append(self.game_logprobs)
        self.values.append(torch.cat(self.game_values, dim=0))

        self.game_states = []
        self.game_actions = []
        self.game_logprobs = []
        self.game_values = []

        return None
    
class MoraOpponent:
    def __init__(self, 
                 name: str, 
                 model: GRUAgent = None, 
                 noise: float = 0.0):
        
        self.name = name
        self.model = model
        self.noise = noise
        self.replay_buffer = []
        self._device = 'cpu'

        if self.model is not None:
            self.next_gru_state = torch.zeros(self.model.gru.num_layers, 1, self.model.gru.hidden_size).to(self._device)

    def _init_state(self, data):
        if data:
            last_round_data = data[-1].copy()
            last_round_data.pop(self.name)
            return [last_round_data[player]['finger'] / max(POSSIBLE_FINGERS) for player in last_round_data.keys()] + \
                   [last_round_data[player]['amount'] / max(POSSIBLE_AMOUNTS) for player in last_round_data.keys()]
        
        return [-1] * STATE_DIM
        
    def get_action(self, data):
        if random.random() < self.noise:
            return {'finger': random.choice(POSSIBLE_FINGERS), 'amount': random.choice(POSSIBLE_AMOUNTS)}
        else:
            next_state = torch.Tensor(self._init_state(data)).to(self._device)
            with torch.no_grad():
                actions, _, _, _, self.next_gru_state = self.model.get_action_and_value(
                    next_state,
                    self.next_gru_state
                )

            return {'finger': POSSIBLE_FINGERS[actions[0]], 'amount': POSSIBLE_AMOUNTS[actions[1]]}

    def set_buffer(self, data):
        # self.replay_buffer.append(data)
        if self.model is not None:
            self.next_gru_state = torch.zeros(self.model.gru.num_layers, 1, self.model.gru.hidden_size).to('cpu')
        return None

class MoraRandom:
    def __init__(self, 
                 name: str):
        self.name = name
        
    def get_action(self, data):
        return {'finger': random.choice(POSSIBLE_FINGERS), 'amount': random.choice(POSSIBLE_AMOUNTS)}

    def set_buffer(self, data):
        # self.replay_buffer.append(data)
        return None