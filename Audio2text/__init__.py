import whisperx
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

token = "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"

# model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="vi")
diarize_model = whisperx.DiarizationPipeline(use_auth_token=token,
                                             device=device)