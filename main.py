import time
import os
import gym
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import wandb
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer
from td3_torch import Agent


if __name__ == '__main__':
    if not os.path.exists('tmp/td3'):
        os.makedirs('tmp/td3')


    env_name = 'Door'
    # config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")

    config = load_composite_controller_config(controller="BASIC")

    env = suite.make(
        env_name="Door",
        robots=["Panda"],
        controller_configs=config,  
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,    
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)
    
    wandb.init(project="TD3-Robosuite", name="td3_door_env")


    actor_learing_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    

    agent = Agent(actor_learning_rate = actor_learing_rate,
                  critic_learning_rate= critic_learning_rate,
                  tau=0.005,
                  input_dims=env.observation_space.shape,
                  env=env,
                  n_actions=env.action_space.shape[0],
                  batch_size=batch_size,
                  layer1_size=layer1_size,
                  layer2_size=layer2_size)
    
    # writer = SummaryWriter("logs")
    n_games = 10000
    best_score = 0
    episode_identifier = f"0-actor_learning_rate={actor_learing_rate}-critic_learning_rate={critic_learning_rate}-batch_size={batch_size}-layer1_size={layer1_size}-layer2_size={layer2_size}"

    agent.load_models()

    for i in range(n_games):
        observation,info = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
    
            next_observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation
        

        wandb.log({
            "Episode": i,
            "Score": score,
            "Actor Learning Rate": actor_learing_rate,
            "Critic Learning Rate": critic_learning_rate,
            "Batch Size": batch_size,
            "Layer1 Size": layer1_size,
            "Layer2 Size": layer2_size
        })

        if i % 10 == 0:
            agent.save_models()

        print(f"Episode {i} Score {score}")

    wandb.finish()  
