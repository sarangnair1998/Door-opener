# TD3 Robotic Manipulation with Robosuite

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Robosuite](https://img.shields.io/badge/Robosuite-0.1.0-blue)](https://robosuite.ai/)

Implementation of Twin Delayed DDPG (TD3) for robotic door opening task using Robosuite's Panda environment. Developed as a learning project for understanding deep reinforcement learning pipelines and robotic manipulation challenges.

## Key Features
- TD3 algorithm implementation with PyTorch
- Robosuite integration for realistic robotic simulation
- Training pipeline with WandB logging
- Visualization script for policy evaluation
- Experience replay buffer with tuple state handling
- Model checkpointing system

## Video of it working


https://github.com/user-attachments/assets/4e2c910c-6881-48ca-bf3a-0fa3ffef9356


## Installation
```bash
# Clone repository
git clone https://github.com/your-username/td3-robosuite-door.git
cd td3-robosuite-door

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Training the Agent
mjpython main.py

## For visualizing
# On macOS (requires mjpython wrapper)
mjpython test.py




