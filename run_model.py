import os

import sys
import time
import torch

from model import ConvNet1
from tminterfaceenv import TmInterfaceEnv
import torchvision

path = sys.argv[2]
tracks = []
if os.path.isdir(path):
    tracks = [path + "\\" + x for x in os.listdir(path) if x.endswith(".Challenge.Gbx")]
else:
    tracks = [path]

if len(tracks) == 0:
    raise RuntimeError("no tracks found")
elif len(tracks) == 1:
    track = tracks[0]
else:
    x = int(input("Select track: \n" + "\n".join([str(i+1) + ": " + os.path.basename(track) for i,track in enumerate(tracks)]) + "\n"))
    track = tracks[x-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env1 = TmInterfaceEnv(sys.argv[1], track, with_state_output=True)
#env2 = TmInterfaceEnv(sys.argv[1], track, with_state_output=True)
model = ConvNet1(480, 640, outputs=4, state_input_size=16).to(device)
model.load_state_dict(torch.load("./train.pth"))
model.eval()

#print(model)
state1 = env1.reset()

while True:
    #now = time.perf_counter()
    screen = torch.from_numpy(state1[0]).to(device).float().unsqueeze(0)
    state = torch.from_numpy(state1[1]).to(device).float().unsqueeze(0)
    inputs = model(screen, state).cpu().detach().numpy()
    #then = time.perf_counter()
    #print(f"model took {(then - now) * 1000}ms")
    state1,_,done,_ = env1.step(inputs)
    if done:
        break