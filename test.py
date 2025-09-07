import torch
import torch.nn as nn
import time
import sys
import os
import argparse

def measure_inference_time(model, dummy_input, iterations=100, device='auto'):
    """
    Measures the average inference time of a PyTorch model, supporting both CPU and GPU.
    """
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    model.to(device)
    if isinstance(dummy_input, dict):
        for key in dummy_input:
            if isinstance(dummy_input[key], torch.Tensor):
                dummy_input[key] = dummy_input[key].to(device)
    else:
        dummy_input = dummy_input.to(device)
    model.eval()

    # Warm-up runs to stabilize performance metrics
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Timing using CUDA events for precision on GPU
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = torch.zeros((iterations,))
        with torch.no_grad():
            for rep in range(iterations):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()  # Wait for the operation to complete
                timings[rep] = starter.elapsed_time(ender)  # Time in milliseconds
        mean_time = timings.mean().item()
        std_time = timings.std().item()
    else:
        # Fallback to standard timing for CPU
        timings = []
        with torch.no_grad():
            for _ in range(iterations):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                timings.append((end_time - start_time) * 1000)
        mean_time = sum(timings) / iterations
        std_time = torch.tensor(timings).std().item()

    return mean_time, std_time



def test_complexsenet(device):
    from models.complexsenet import ComplexSENet
    from models.complexsenet_pt import ComplexSENet as ComplexSENetPT

    models_to_test = {
        "ComplexSENetPT": ComplexSENetPT,
        "ComplexSENet": ComplexSENet,
    }

    dummy_input = torch.randn(2, 201, 200, 2)
    
    print(f"--- Starting ComplexSENet Inference Time Benchmark on {device.upper()} ---")

    for name, model_class in models_to_test.items():
        try:
            model = model_class(
                dense_channel=64,
                beta=2.0,
                num_tsblock=4
            )
            
            mean_ms, std_ms = measure_inference_time(model, dummy_input, device=device)
            
            print(f"Model: {name:<15} | Avg Inference Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
        
        except Exception as e:
            print(f"Error testing model {name}: {e}")
    print("--- Benchmark Finished ---")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Model inference speed test')
    parser.add_argument('--cpu', action='store_true', help='Force CPU testing even if GPU is available')
    args = parser.parse_args()
    
    # test_primeKnetv5()
    # test_primeKnetv7()
    
    # Determine device based on arguments
    if args.cpu:
        device = 'cpu'
        print("Forcing CPU testing as requested with --cpu flag")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_complexsenet(device)