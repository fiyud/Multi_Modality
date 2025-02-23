import time
import numpy as np
from Denoiser.denoiser import run
import onnxruntime
import librosa
import scipy
import os
from Denoiser import *

def denoise_audio(audio_link, output_folder='',session = session, format_type = 'wav'):
    wav, sr = librosa.load(audio_link, mono=True)

    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 4

    start = time.time()

    wav_onnx, new_sr = run(session, wav, sr, batch_process_chunks=False)

    print(f'Ran in {time.time() - start}s')

    # output_path = output_folder + audio_link.split('/')[-1].split('.')[0] + '_denoised.' + format_type 
    output_path = os.path.join(output_folder, audio_link.split('\\')[-1].split('.')[0] + '_denoised.' + format_type)
    
    scipy.io.wavfile.write(output_path, new_sr, wav_onnx)

    return output_path