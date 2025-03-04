import time
import os
import gym
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from td3_torch import Agent

if __name__ == '__main__':
    config = load_composite_controller_config(controller="BASIC")

    env = suite.make(
        env_name="Door",
        robots=["Panda"],
        controller_configs=config,
        has_renderer=True,  
        use_camera_obs=False,
        horizon=300,    
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate,
                  critic_learning_rate=critic_learning_rate,
                  tau=0.005,
                  input_dims=env.observation_space.shape,
                  env=env,
                  n_actions=env.action_space.shape[0],
                  batch_size=batch_size,
                  layer1_size=layer1_size,
                  layer2_size=layer2_size)
    
    agent.load_models()

    num_episodes = 5  
    for i in range(num_episodes):
        observation, info = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, validation=True) 
            next_observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            env.render()  
            
            score += reward
            observation = next_observation
            time.sleep(0.03)  

        print(f"Episode {i} Score: {score}")

