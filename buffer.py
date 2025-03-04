import numpy as np


class ReplayBuffer:
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size,*input_shape))
        self.new_state_memory = np.zeros((self.mem_size,*input_shape))
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype=bool)
        
    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        # print(f"DEBUG: Type of state: {type(state)}")
        # print(f"DEBUG: State content: {state}")
        # print(f"DEBUG: State element types: {[type(s) for s in state]}")


        # Convert tuple to NumPy array
        if isinstance(state, tuple):
            state = np.concatenate([np.ravel(s) for s in state], dtype=np.float32)

        if isinstance(next_state, tuple):
            next_state = np.concatenate([np.ravel(s) for s in next_state], dtype=np.float32)

        # print(f"DEBUG: Converted State Shape: {state.shape}")

        self.state_memory[index] = state
        
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

        
    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr,self.mem_size)
        
        batch = np.random.choice(max_mem,batch_size,replace=False)
        
        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states,actions,rewards,next_states,terminal
        