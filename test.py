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
    for key in dummy_input:
        if isinstance(dummy_input[key], torch.Tensor):
            dummy_input[key] = dummy_input[key].to(device)
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

def test_primeKnetv7():
    from models.primeKnetv7 import PrimeKnetv7
    model = PrimeKnetv7(
        fft_len=400,
        dense_channel=64,
        num_tsblock=4
    )
    model.eval()
    model.to("cuda")
    dummy_input = {
        "magnitude": torch.randn(1, 201, 400).to("cuda"),
        "phase": torch.randn(1, 201, 400).to("cuda")
    }
    y = model(dummy_input)
    print(y.shape)

def test_primeKnetv5():
    from models.primeKnetv5 import PrimeKnetv5
    model = PrimeKnetv5(
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=2,
        num_tsblock=4
    )
    model.eval()
    model.to("cuda")
    dummy_input = {
        "magnitude": torch.randn(1, 201, 400).to("cuda"),
        "phase": torch.randn(1, 201, 400).to("cuda")
    }
    y = model(dummy_input)
    # print(y.shape)  

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Model inference speed test')
    parser.add_argument('--cpu', action='store_true', help='Force CPU testing even if GPU is available')
    args = parser.parse_args()
    
    # test_primeKnetv5()
    test_primeKnetv7()
    '''
    # Add the project's root directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    
    models_to_test = {}

    # Statically import each model and add it to the test dictionary
    # This approach is robust to missing model files
    try:
        from models.primeKnet import PrimeKnet
        models_to_test["PrimeKnet"] = PrimeKnet
    except ImportError:
        print("Skipping PrimeKnet: Not found.")
    # try:
    #     from models.primeKnetv3 import PrimeKnetv3
    #     models_to_test["PrimeKnetv3"] = PrimeKnetv3
    # except ImportError:
    #     print("Skipping PrimeKnetv3: Not found.")

    # try:
    #     from models.primeKnetv4 import PrimeKnetv4
    #     models_to_test["PrimeKnetv4"] = PrimeKnetv4
    # except ImportError:
    #     print("Skipping PrimeKnetv4: Not found.")
    # try:
    #     from models.primeKnetv5 import PrimeKnetv5
    #     models_to_test["PrimeKnetv5"] = PrimeKnetv5
    # except ImportError:
    #     print("Skipping PrimeKnetv5: Not found.")
    try:
        from models.primeKnetv7 import PrimeKnetv7
        models_to_test["PrimeKnetv7"] = PrimeKnetv7
    except ImportError:
        print("Skipping PrimeKnetv7: Not found.")

    if not models_to_test:
        print("No models were found to test. Make sure the model files exist.")
    else:
        # Determine device based on arguments
        if args.cpu:
            device = 'cpu'
            print("Forcing CPU testing as requested with --cpu flag")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"--- Starting Inference Time Benchmark on {device.upper()} ---")

        for name, model_class in models_to_test.items():
            try:
                model = model_class(
                    fft_len=400,
                    dense_channel=64,
                    sigmoid_beta=2,
                    num_tsblock=4
                )
                
                dummy_input = {
                    'magnitude': torch.randn(1, 201, 400),
                    'phase': torch.randn(1, 201, 400)
                }
                
                mean_ms, std_ms = measure_inference_time(model, dummy_input, device=device)
                
                print(f"Model: {name:<12} | Avg Inference Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
            
            except Exception as e:
                print(f"Error testing model {name}: {e}")

        print("--- Benchmark Finished ---")
'''