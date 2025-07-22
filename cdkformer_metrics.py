import torch
import time
from fvcore.nn import FlopCountAnalysis, parameter_count
from models.CDKformer import conformermulti

def calculate_cdkformer_metrics():
    """
    Calculate CDKFormer metrics using fvcore
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and set to eval mode
    model = conformermulti(num_classes=36, ind=0).to(device)
    model.eval()
    
    input_size = (1, 1, 12, 51)
    dummy_input = torch.randn(*input_size).to(device)
    
    try:
        # Calculate parameters
        params = parameter_count(model)
        total_params = sum(params.values())
        
        # Calculate FLOPs
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        
        # Calculate memory usage
        mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        mem_total = mem_params + mem_bufs
        
        # Calculate inference time
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)
            
            # Measure time
            start_time = time.time()
            for _ in range(100):
                _ = model(dummy_input)
            end_time = time.time()
        
        inference_time = (end_time - start_time) / 100
        fps = 1 / inference_time
        
        # Print results
        print("\nCDKFormer Metrics:")
        print("="*50)
        print(f"Total parameters: {total_params}")
        print(f"Parameters size (MB): {mem_params / (1024 ** 2):.2f}")
        print(f"Total memory (MB): {mem_total / (1024 ** 2):.2f}")
        print(f"FLOPs (GFLOPs): {total_flops / (10**9):.2f}")
        print(f"FPS: {fps:.2f}")
    
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        print("Note: FLOPs calculation may not work for Conformer architecture")

if __name__ == '__main__':
    calculate_cdkformer_metrics()