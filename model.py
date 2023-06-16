import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, version=1):
        super(Net, self).__init__()
        self.version = version

        # Check version and apply appropriate transformations
        if self.version == 1:
            # Version 1 - Lighter model without Batch Normalization (BN), Dropout (DP), and Global Average Pooling (GAP)
            # Here, each convolution layer only includes ReLU activation
            # After each convolution layer, the receptive field (RF) is commented
            self.model = nn.Sequential(
                nn.Conv2d(1, 8, 3), # RF: 3
                nn.ReLU(),

                nn.Conv2d(8, 10, 3), # RF: 5
                nn.ReLU(),

                nn.Conv2d(10, 10, 3), # RF: 7
                nn.ReLU(),

                nn.MaxPool2d(2, 2), # RF: 14

                nn.Conv2d(10, 10, 1), # RF: 14
                nn.ReLU(),

                nn.Conv2d(10, 10, 3), # RF: 16
                nn.ReLU(),

                nn.Conv2d(10, 10, 3), # RF: 18
                nn.ReLU(),

                nn.Conv2d(10, 10, 1), # RF: 18
                nn.ReLU(),

                nn.Conv2d(10, 10, 7), # RF: 24
            )
        elif self.version == 2:
            # Version 2 - With Batch Normalization (BN)
            # BN layers are added after each convolution layer to standardize feature maps
            # RF is updated with each layer accordingly
            self.model = nn.Sequential(
                nn.Conv2d(1, 8, 3), # RF: 3
                nn.BatchNorm2d(8),
                nn.ReLU(),

                nn.Conv2d(8, 10, 3), # RF: 5
                nn.BatchNorm2d(10),
                nn.ReLU(),

                nn.Conv2d(10, 10, 3), # RF: 7
                nn.BatchNorm2d(10),
                nn.ReLU(),

                nn.MaxPool2d(2, 2), # RF: 14

                nn.Conv2d(10, 10, 1), # RF: 14
                nn.BatchNorm2d(10),
                nn.ReLU(),

                nn.Conv2d(10, 10, 3), # RF: 16
                nn.BatchNorm2d(10),
                nn.ReLU(),

                nn.Conv2d(10, 10, 3), # RF: 18
                nn.BatchNorm2d(10),
                nn.ReLU(),

                nn.Conv2d(10, 10, 1), # RF: 18
                nn.BatchNorm2d(10),
                nn.ReLU(),

                nn.Conv2d(10, 10, 7), # RF: 24
            )
        elif self.version == 3:
            # Version 3 - With BN and DP
            # Added Dropout after each convolution block
            self.model = nn.Sequential(
                nn.Conv2d(1, 8, 3), # RF: 3
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Conv2d(8, 10, 3), # RF: 5
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Conv2d(10, 10, 3), # RF: 7
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.MaxPool2d(2, 2), # RF: 14

                nn.Conv2d(10, 10, 1), # RF: 14
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Conv2d(10, 10, 3), # RF: 16
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Conv2d(10, 10, 3), # RF: 18
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Conv2d(10, 10, 1), # RF: 18
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Conv2d(10, 10, 7), # RF: 24
            )
        elif self.version == 4:
            # Version 4 - Original model optimized with BN, DP and GAP
            # Added GAP layer before the final convolution
            self.model = nn.Sequential(
                nn.Conv2d(1, 8, 3, bias=False), # RF: 3
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Dropout(0.1),

                nn.Conv2d(8, 10, 3, bias=False), # RF: 5
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Dropout(0.1),

                nn.Conv2d(10, 12, 1, bias=False), # RF: 5
                nn.MaxPool2d(2, 2), # RF: 10

                nn.Conv2d(12, 14, 3, bias=False), # RF: 12
                nn.ReLU(),
                nn.BatchNorm2d(14),
                nn.Dropout(0.1),

                nn.Conv2d(14, 16, 3, bias=False), # RF: 14
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Dropout(0.1),

                nn.Conv2d(16, 8, 3, bias=False), # RF: 16
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Dropout(0.1),

                nn.Conv2d(8, 12, 3, padding=1, bias=False), # RF: 18
                nn.ReLU(),
                nn.BatchNorm2d(12),
                nn.Dropout(0.1),

                nn.AvgPool2d(6), # RF: 23
                nn.Conv2d(12, 10, 1, bias=False), # RF: 23
            )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
