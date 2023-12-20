import os
import torch
import cupy as cp


def setting_os_path(path):
    current_path = os.getcwd()
    if current_path != path:
        os.chdir(path)
        return print("** Path ** \n", "From ", current_path, "\nTo", path)
    return print("Current Path: ", path)


def get_device(bus_id, cuda_id):
    os.environ["CUDA_DEVICE_ORDER"] = bus_id  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id  # Set the GPU 2 to use
    cp.cuda.Device(cuda_id).use()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    return device
