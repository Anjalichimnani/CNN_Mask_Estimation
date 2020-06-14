from imports.imports_eva import *

class CustomNet (nn.Module):

  def __init__ (self):
    super(CustomNet, self).__init__()

    self.conv_block_1 = nn.Sequential (
        nn.Conv2d (in_channels=6, out_channels=64, kernel_size= 3, bias = False),
        nn.BatchNorm2d (64),
        nn.ReLU()
    )

    self.conv_block_2 = nn.Sequential (
        nn.Conv2d (in_channels=64, out_channels=128, kernel_size=3),
        nn.BatchNorm2d (num_features=128),
        nn.ReLU()
    )

    self.pool1 = nn.MaxPool2d (kernel_size=2, stride=2)

    self.conv_block_3 = nn.Sequential (
        nn.Conv2d (in_channels=128, out_channels=128, kernel_size=3),
        nn.BatchNorm2d (num_features=128),
        nn.ReLU()
    )

    self.conv_block_4 = nn.Sequential (
        nn.Conv2d (in_channels=128, out_channels=256, kernel_size=3),
        nn.BatchNorm2d (num_features=256),
        nn.ReLU()
    )

    self.conv_block_5 = nn.Sequential (
        nn.Conv2d (in_channels=256, out_channels=1, kernel_size=3)
    )

  def forward (self, x):
    x = self.conv_block_1 (x)
    x = self.conv_block_2 (x)

    x = self.pool1 (x)

    x = self.conv_block_3 (x)
    x = self.conv_block_4 (x)
    x = self.conv_block_5 (x)

    return x