import torch
from Speech_emo_2.model import Dual
model = Dual()
# model.load_state_dict(torch.load(r'D:\NCKHSV.2024-2025\Services\aidemo\Speech_emo_2\models\emopre.pth', map_location='cpu'))
model.load_state_dict(torch.load(r"C:\Users\ADMIN\Downloads\best_model_0.pth", map_location='cuda'))
