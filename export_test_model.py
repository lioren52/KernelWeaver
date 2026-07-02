import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches the dimensions of the user's manual Graph
        self.fc1 = nn.Linear(64, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x

if __name__ == "__main__":
    model = SimpleModel()
    
    # Batch size 128, Input dim 64
    dummy_input = torch.randn(128, 64)
    
    onnx_path = "test_model.onnx"
    print(f"Exporting PyTorch model to {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=["X"], 
        output_names=["Y"],
        opset_version=14
    )
    print("Done! Now run:")
    print(f"python onnx_exporter.py {onnx_path}")
