from typing import List

import numpy as np
import torch
from tminterface.structs import SimStateData
from torch import nn

def sim_state_to_state_input(index, states : List[SimStateData]):
    current_state: SimStateData = states[min(index, len(states) - 1)]
    current_input_state = np.array(
        [current_state.input_accelerate, current_state.input_brake, current_state.input_left,
         current_state.input_right])
    global_to_local_rot = np.transpose(np.array(current_state.rotation_matrix))
    position = np.array(current_state.position)
    velocity = global_to_local_rot.dot(np.array(current_state.velocity))
    position_in_1_sec = global_to_local_rot.dot(np.array(
        states[min(index + 100, len(states) - 1)].position) - position)
    position_in_2_sec = global_to_local_rot.dot(np.array(
        states[min(index + 200, len(states) - 1)].position) - position)
    position_in_5_sec = global_to_local_rot.dot(np.array(
        states[min(index + 500, len(states) - 1)].position) - position)
    # print(f"track_index={track_index} item={item} time_ms={time_ms} inputs={inputs}")
    # print(f"frame.shape={frame.shape}")

    state = np.concatenate([current_input_state, velocity, position_in_1_sec, position_in_2_sec, position_in_5_sec], axis=0)
    # print("velocity=",velocity)
    # print("position_in_1_sec=",position_in_1_sec)
    # print("position_in_2_sec=",position_in_2_sec)
    # print("position_in_5_sec=",position_in_5_sec)
    # print("state=", state)
    return state


class ConvNet1(nn.Module):

    def __init__(self, h, w, outputs, state_input_size):
        super(ConvNet1, self).__init__()
        self.max_pooling = [2,2,3]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w) // self.max_pooling[0]) // self.max_pooling[1]) // self.max_pooling[2]
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h) // self.max_pooling[0]) // self.max_pooling[1]) // self.max_pooling[2]
        linear_input_size = convw * convh * 32 + state_input_size
        #print(f"linear_input_size={linear_input_size}")
        if state_input_size != 0:
            self.stateMixer = nn.Linear(state_input_size, state_input_size)
        self.head1 = nn.Linear(linear_input_size, linear_input_size // 2)
        self.head2 = nn.Linear(linear_input_size // 2, linear_input_size // 4)
        self.head3 = nn.Linear(linear_input_size // 4, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, state = None):
        if self.stateMixer and state is None:
            raise RuntimeError("ConvNet1.forward state input must be set")
        x = torch.relu(torch.max_pool2d(self.bn1(self.conv1(x)), self.max_pooling[0]))
        x = torch.relu(torch.max_pool2d(self.bn2(self.conv2(x)), self.max_pooling[1]))
        x = torch.relu(torch.max_pool2d(self.bn3(self.conv3(x)), self.max_pooling[2]))
        if state is not None:
            state = torch.tanh(self.stateMixer(state))
            x = torch.cat((state, x.view(x.size(0), -1)), 1)
        else:
            x = x.view(x.size(0), -1)

        x = torch.relu(self.head1(x))
        x = torch.relu(self.head2(x))
        x = self.head3(x)
        return 0.5 * torch.tanh(x) + 0.5


class DDPGActorCritic:
    def __init__(self, obs_space, action_space):
        self.pi = ConvNet1(480, 640, 4, 0)
        self.q = ConvNet1(480, 640, 1, 4)

    def act(self, observation):
        return self.pi(observation).cpu().detach().squeeze().numpy()

    def parameters(self):
        return [*self.pi.parameters(), *self.q.parameters()]