import torch

def test_is_cuda():
    assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation."
if __name__ == "__main__":
    test_is_cuda()
    print("CUDA is available. Test passed.")