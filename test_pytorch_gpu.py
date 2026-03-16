import torch

def test_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: No GPU detected. CUDA is not available.")
        return

    device_count = torch.cuda.device_count()
    print(f"GPU count: {device_count}")

    for i in range(device_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")

    # Run a simple tensor operation on the GPU
    device = torch.device("cuda:0")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

    print(f"\nTensor device: {c.device}")
    print(f"Matrix multiply result shape: {c.shape}")
    print(f"Result sample value: {c[0, 0].item():.4f}")
    print("\nPASS: GPU is working correctly.")

if __name__ == "__main__":
    test_gpu()
