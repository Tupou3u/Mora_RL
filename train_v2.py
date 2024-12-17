from mora_env import Mora
from mora_player import *
from mora_params import *
from algs.init_model_gru import GRUAgent
from algs.ppo_gru_torch import PPO_GRU
import torch
import time
import random
from utils import *

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

save_path = 'history/train_2/train_2_3/'

agent = PPO_GRU(
    learning_rate=0.0001,    
    batch_size=100,
    train_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    actor_clip=0.1,
    critic_clip=0.1,
    ent_coef=0.0,          
    alpha=0.1,                    
    device=train_device,
    log_dir=save_path
)

game = Mora()

NUM_EPOCHS = 1_000_000
NUM_GAMES = 1_000
VERBOSE = 100
LR_VERBOSE = 5
LR_DECAY = 0.95
MIN_LR = 1e-5

start = time.time()
total_rounds = 0

last_ep_reward = [-100, 1]

for epoch in range(1, NUM_EPOCHS):
    ep_start_time = time.time()
    curr_model = copy_model(agent.model, rollout_device)

    player = MoraPlayer('player', model=curr_model)
    game.add_player(player)
    for num, opp_model in enumerate(range(NUM_PLAYERS-1)):
        game.add_player(MoraRandom(f'rnd_{num+1}'))
    
    game.play_games(NUM_GAMES)
    game.kick_all()

    total_rounds += NUM_GAMES * NUM_ROUNDS
    _ep_reward = torch.sum(torch.cat(player.rewards)).item() / NUM_GAMES
    ep_reward = [_ep_reward, epoch]

    if agent.log_dir:
        agent.writer.add_scalar("rollout/ep_reward", _ep_reward, agent.num_trains)

    # print([round(r, 3) for r in mean_rewards])
    print(f"epoch: {epoch} total_rounds: {total_rounds} episode_reward: {round(_ep_reward, 3)} time: {round(time.time() - start, 0)}s")
    
    if ep_reward[0] >= last_ep_reward[0]:
        print(f'New best model!')
        curr_model_state = agent.model.state_dict()
        if agent.log_dir:
            path = agent.log_dir + 'best_agent.pt'
            torch.save(curr_model_state, path)
        last_ep_reward = ep_reward

    if (ep_reward[1] - last_ep_reward[1]) % LR_VERBOSE == 0 and (ep_reward[1] - last_ep_reward[1]) != 0:
        agent.learning_rate = max(MIN_LR, LR_DECAY * agent.learning_rate)
        agent.optimizer.param_groups[0]["lr"] = agent.learning_rate
    
        if agent.log_dir:
            agent.writer.add_scalar("params/lr", agent.learning_rate, agent.num_trains)

        print(f'Learning rate update: {agent.learning_rate}')

    if ep_reward[1] - last_ep_reward[1] == VERBOSE:
        print('End of train')
        break
    else:
        agent.train(
            player.states, 
            player.actions, 
            player.rewards, 
            player.logprobs, 
            player.values
        )

        if agent.log_dir:
            agent.writer.add_scalar("ep_data/ep_time", time.time() - ep_start_time, agent.num_trains)

