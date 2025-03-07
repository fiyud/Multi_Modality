import onnxruntime

model_link = r"D:\NCKHSV.2024-2025\Services\aidemo\Denoiser\models\denoiser.onnx"
provider = "CPUExecutionProvider" if onnxruntime.get_device() == "CPU" else "CUDAExecutionProvider"

opts = onnxruntime.SessionOptions()
opts.inter_op_num_threads = 4
opts.intra_op_num_threads = 4
opts.log_severity_level = 4

session = onnxruntime.InferenceSession(
        model_link,
        providers=[provider],
        sess_options=opts,
    )