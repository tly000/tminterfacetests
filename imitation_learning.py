import bisect
import os
import pickle
import random

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from tminterface.structs import SimStateData
from decord import VideoReader

from model import ConvNet1, sim_state_to_state_input


class TrackmaniaDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.tracks = []
        if os.path.isdir(path):
            self.tracks = [path + "/" + x for x in os.listdir(path) if x.endswith(".Challenge.Gbx")]
        else:
            self.tracks = [path]

        self.video_loaders = [VideoReader(x + ".mp4") for x in self.tracks]
        self.states_per_track = [pickle.load(open(x + ".states.bin", "rb")) for x in self.tracks]
        self.offset_per_track = [0]
        for states in self.states_per_track:
            self.offset_per_track.append(self.offset_per_track[-1] + len(states))

    def __len__(self):
        return self.offset_per_track[-1]

    def __getitem__(self, item):
        track_index = bisect.bisect_right(self.offset_per_track, item) - 1
        item -= self.offset_per_track[track_index]
        next_state: SimStateData = self.states_per_track[track_index][
            min(item + 1, len(self.states_per_track[track_index]) - 1)]
        # assert state.race_time == 10 * item
        inputs = [next_state.input_accelerate, next_state.input_brake, next_state.input_left, next_state.input_right]
        image_path = os.path.dirname(self.tracks[track_index]) + "/images/" + os.path.basename(
            self.tracks[track_index]) + "_" + str(item) + ".jpg"
        if os.path.isfile(image_path):
            frame = torchvision.io.read_image(image_path)
        else:
            # print(image_path)
            frame = self.video_loaders[track_index][item]
            # TODO this seems inefficient...
            frame = torch.transpose(torch.transpose(torch.tensor(frame.asnumpy()), 0, 2), 1, 2)
            torchvision.io.write_jpeg(frame, image_path)

        # print(f"track_index={track_index} item={item} time_ms={time_ms} inputs={inputs}")
        # print(f"frame.shape={frame.shape}")
        return frame.float(), torch.tensor(sim_state_to_state_input(item, self.states_per_track[track_index])).float(), torch.tensor(inputs).float()


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = ConvNet1(480, 640, outputs=4, state_input_size=16).to(device)
# print(net)
if os.path.isfile("./train.pth"):
    net.load_state_dict(torch.load("./train.pth"))
    net.eval()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 100
batch_size = 64
trainset = TrackmaniaDataset(sys.argv[2])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

net.train()

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, state, expected_output = data
        # torchvision.io.write_jpeg(inputs[0,:,:,:].byte(), "test.jpg")
        # exit()
        inputs, state, expected_output = inputs.to(device), state.to(device), expected_output.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs, state)

        # print(torch.sum((outputs > 0.5) == expected_output) / torch.numel(labels) * 100)

        loss = criterion(outputs, expected_output)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

    torch.save(net.state_dict(), "./train.pth")

print('Finished Training')
