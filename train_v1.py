from mora_env import Mora
from mora_player import MoraPlayer, MoraOpponent
from mora_params import *
from algs.init_model_gru import GRUAgent
from algs.ppo_gru_torch import PPO_GRU
import torch
import time
import random

def copy_model(model: GRUAgent, device):
    state_dict = model.state_dict()
    model_copy = GRUAgent(device).to(device)
    model_copy.load_state_dict(state_dict)
    torch.compile(model_copy)
    return model_copy

def load_model(path, device):
    model = GRUAgent(device).to(device)
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)
    return model

rollout_device = 'cpu'
train_device = 'cuda'

save_path = 'history/train_1_2/'

agents = [
    PPO_GRU(
        learning_rate=0.0001,    
        batch_size=100,
        train_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        actor_clip=0.1,
        critic_clip=0.1,
        ent_coef=0.01,          
        alpha=0.1,                    
        device=train_device,
        log_dir=save_path + 'agent_1/'
    ),
    PPO_GRU(
        learning_rate=0.0001,           
        batch_size=100,
        train_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        actor_clip=0.1,
        critic_clip=0.1,
        ent_coef=0.01,          
        alpha=0.1,                    
        device=train_device,
        log_dir=save_path + 'agent_2/'
    ),
    PPO_GRU(
        learning_rate=0.0001,          
        batch_size=100,
        train_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        actor_clip=0.1,
        critic_clip=0.1,
        ent_coef=0.01,          
        alpha=0.1,                    
        device=train_device,
        log_dir=save_path + 'agent_3/'
    ),
    PPO_GRU(
        learning_rate=0.0001,          
        batch_size=100,
        train_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        actor_clip=0.1,
        critic_clip=0.1,
        ent_coef=0.01,          
        alpha=0.1,                    
        device=train_device,
        log_dir=save_path + 'agent_4/'
    )
]

game = Mora()

NUM_EPOCHS = 1_000_000
NUM_GAMES = 1_000
VERBOSE = 10
LR_VERBOSE = 5
LR_DECAY = 0.5
MIN_LR = 0.0
LAST_LR = 0.0001
ENT_DECAY = 0.95
MIN_ENT = 0.001
ALPHA = 0.0 
ALPHA_DECAY = 0.95
MIN_ALPHA = 0.0

start = time.time()
total_rounds = 0

last_ep_reward = [-100, 1]
cycle_rewards_dict = {}

current_cycle = agents[:]
random.shuffle(current_cycle)
curr_agent = current_cycle.pop()
cycle_count = 1

print(f'Start train of agent {agents.index(curr_agent)+1}')

for epoch in range(1, NUM_EPOCHS):
    ep_start_time = time.time()
    curr_model = copy_model(curr_agent.model, rollout_device)
    opponents = agents[:]
    opponents.pop(agents.index(curr_agent))
    opponents_models = [copy_model(opp.model, rollout_device) for opp in opponents]

    curr_player = MoraPlayer('player', model=curr_model)
    game.add_player(curr_player)
    for num, opp_model in enumerate(opponents_models):
        game.add_player(MoraOpponent(f'opponent_{num}', model=opp_model, noise=ALPHA))
    
    game.play_games(NUM_GAMES)
    game.kick_all()

    total_rounds += NUM_GAMES * NUM_ROUNDS
    curr_agent_num = agents.index(curr_agent)
    _ep_reward = torch.sum(torch.cat(curr_player.rewards)).item() / NUM_GAMES
    ep_reward = [_ep_reward, epoch]

    if curr_agent.log_dir:
        curr_agent.writer.add_scalar("rollout/ep_reward", _ep_reward, curr_agent.num_trains)

    # print([round(r, 3) for r in mean_rewards])
    print(f"epoch: {epoch} total_rounds: {total_rounds} episode_reward: {round(_ep_reward, 3)} time: {round(time.time() - start, 0)}s")
    
    if ep_reward[0] >= last_ep_reward[0]:
        print(f'New best model!')
        cycle_rewards_dict[curr_agent] = _ep_reward
        curr_model_state = curr_agent.model.state_dict()
        if curr_agent.log_dir:
            curr_path = curr_agent.log_dir + f'{cycle_count}_agent.pt'
            torch.save(curr_model_state, curr_path)
        last_ep_reward = ep_reward

    if (ep_reward[1] - last_ep_reward[1]) % LR_VERBOSE == 0 and (ep_reward[1] - last_ep_reward[1]) != 0:
        curr_agent.learning_rate = max(MIN_LR, LR_DECAY * curr_agent.learning_rate)
        curr_agent.optimizer.param_groups[0]["lr"] = curr_agent.learning_rate
    
        if curr_agent.log_dir:
            curr_agent.writer.add_scalar("params/lr", curr_agent.learning_rate, curr_agent.num_trains)

        print(f'Learning rate update (agent: {curr_agent_num+1}): {curr_agent.learning_rate}')

    if ep_reward[1] - last_ep_reward[1] == VERBOSE:
        curr_agent.learning_rate = LAST_LR
        curr_agent.optimizer.param_groups[0]["lr"] = curr_agent.learning_rate
        curr_agent.ent_coef = max(MIN_ENT, curr_agent.ent_coef * ENT_DECAY)

        if curr_agent.log_dir:
            curr_agent.writer.add_scalar("params/ent_coef", curr_agent.ent_coef, curr_agent.num_trains)

        print(f'Entropy coef update (agent: {curr_agent_num+1}): {curr_agent.ent_coef}')

        curr_agent.model.load_state_dict(curr_model_state)

        if curr_agent.log_dir:
            curr_agent.writer.add_scalar("cycle/cycle_reward", cycle_rewards_dict[curr_agent], cycle_count)

        if not current_cycle:
            ALPHA = max(MIN_ALPHA, ALPHA * ALPHA_DECAY)
            print(f'Alpha update: {ALPHA}') 
            print(f'Cycles count: {cycle_count}')
            current_cycle = agents[:]
            random.shuffle(current_cycle)
            cycle_rewards_dict = {}
            cycle_count += 1

        curr_agent = current_cycle.pop()
        last_ep_reward = [-100, epoch]
        print(f'\nStart train of agent {agents.index(curr_agent)+1}')
    else:
        curr_agent.train(
            curr_player.states, 
            curr_player.actions, 
            curr_player.rewards, 
            curr_player.logprobs, 
            curr_player.values
        )

        if curr_agent.log_dir:
            curr_agent.writer.add_scalar("ep_data/ep_time", time.time() - ep_start_time, curr_agent.num_trains)

