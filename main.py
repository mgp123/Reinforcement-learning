from pytorch_decorators import pytorch_use_numpy_as_input
import gym
import torch
import torch.nn as nn

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


@pytorch_use_numpy_as_input
class CartPoleQNet(nn.Module):
    def __init__(self):
        super(CartPoleQNet, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #environment = gym.make("CartPole-v1")
    print_hi(42)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
