import ctypes
import os

# 加載cuDNN庫
cudnn_path = "C:\\CUDA_CuDNN\\bin\\cudnn_bin\\cudnn64_8.dll"
if os.path.exists(cudnn_path):
    _ = ctypes.WinDLL(cudnn_path)
    print("cuDNN 加載成功")
else:
    print("cuDNN 加載失敗，請檢查路徑是否正確")

# 測試TensorFlow是否加載了cuDNN
import tensorflow as tf

print("Is TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

def test_gpu():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("請安裝 GPU 版本的 TensorFlow")

test_gpu()
