import os
import torch
import torch.nn.functional
import torch.optim as optim
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(torch.nn.Module):
    def __init__(self, input_dims, n_actions,fc1_dims=256,fc2_dims=128,name='critic',
                 checkpoint_dir='tmp/td3',learning_rate=10e-3):
        super(CriticNetwork,self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'_td3')

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims,1)

        self.optimizer = optim.AdamW(self.parameters(),lr=learning_rate,weight_decay=0.005)

        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

        print(f"Critic Network Initialized on {self.device}")

        self.to(self.device)
        
    def forward(self,state,action):
        action_value = self.fc1(torch.cat([state,action],dim=1))
        action_value = torch.nn.functional.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = torch.nn.functional.relu(action_value)

        q1 = self.q1(action_value)

        return q1
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(),self.checkpoint_file)


    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))

    

class ActorNetwork(torch.nn.Module):

    def __init__(self,input_dims,n_actions=2,fc1_dims=256,fc2_dims=128,name='actor',
                 checkpoint_dir='tmp/td3',learning_rate=10e-3):
        
        super(ActorNetwork,self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+'_td3')

        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims,self.n_actions)


        self.optimizer = optim.Adam(self.parameters(),lr=learning_rate)
        

        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

        print(f"Actor Network Initialized on {self.device}")

        self.to(self.device)

    def forward(self,state):
        x = self.fc1(state)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)

        x = torch.tanh(self.mu(x))

        return x
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(),self.checkpoint_file)

    
    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))











