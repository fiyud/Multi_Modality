from Speech_emo_2.common import CBAM
from torch import nn

class Dual(nn.Module):
    def __init__(self):
        super(Dual, self).__init__()

        self.feature_extractor2 = nn.Sequential(
            # 1st Conv Layer + BatchNorm + ReLU + Pooling
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),  # (128, 251) -> (128, 251, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (128, 251, 64) -> (64, 125, 64)

            # 2nd Conv Layer + BatchNorm + ReLU + Pooling
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),  # (64, 125, 64) -> (64, 125, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),  # (64, 125, 64) -> (16, 31, 64)

            # 3rd Conv Layer + BatchNorm + ReLU + Pooling
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),  # (16, 31, 64) -> (16, 31, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),  # (16, 31, 128) -> (4, 7, 128)

            # 4th Conv Layer + BatchNorm + ReLU + Pooling
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),  # (4, 7, 128) -> (4, 7, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))  # (4, 7, 128) -> (1, 1, 128)
        )

        # Fully Connected Layer
        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),  # 1*1*128 -> 256
            nn.ReLU(),
            nn.Linear(512, 5),  # 256 -> 5
        )

        # self.truefc = nn.Softmax(dim=1)
        self.gru = nn.GRU(input_size=128, hidden_size=256, bidirectional=False, batch_first=True)
        self.fc3 = nn.Linear(5, 5)
        self.cbam = CBAM(128)

    def forward(self, mfcc):
        # wave_form = self.feature_extractor1(wave_form)  # Pass through the sequential feature extractor

        # # LSTM expects input of shape (batch_size, seq_len, input_size)
        # wave_form = wave_form.permute(0, 2, 1)  # Reshape to (batch_size, seq_len, input_size)
        # wave_form, _ = self.lstm(wave_form)

        # wave_form = wave_form[:, -1, :]

        # wave_form = self.fc1(wave_form)

        mfcc = self.feature_extractor2(mfcc)
        mfcc = self.cbam(mfcc)
      
        if mfcc.shape[0] != 1:
            mfcc = mfcc.squeeze().unsqueeze(1)
    
        elif mfcc.shape[0] == 1:
            mfcc = mfcc.squeeze().unsqueeze(0).unsqueeze(0)
       
        mfcc,_ = self.gru(mfcc)
        mfcc = self.fc2(mfcc)
        # x = torch.cat((wave_form, mfcc), dim=1)

        mfcc = self.fc3(mfcc)

        return mfcc
