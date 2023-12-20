import torch.nn as nn

class FC_Classifier(nn.Module):
    def __init__(self,
                 in_channels :int,
                 hid_channels :int,
                 out_channels :int,
                 dropout = 0.1
                ):
        super(FC_Classifier, self).__init__()

        self.fc1 = nn.Linear(in_channels, hid_channels)
        self.fc2 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.CELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))