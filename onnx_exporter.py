import onnx
from onnx import numpy_helper
import sys
import numpy as np
import os

def export_onnx(onnx_path, graph_path):
    print(f"Loading {onnx_path}...")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # Extract weights (initializers)
    initializers = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        initializers[init.name] = arr
        
    # Extract inputs
    inputs = {}
    for inp in graph.input:
        if inp.name not in initializers:
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            # Handle dynamic batch size by defaulting to 128 (our benchmark size)
            if not isinstance(shape[0], int) or shape[0] <= 0:
                shape[0] = 128
            inputs[inp.name] = shape
            
    with open(graph_path, 'w') as f:
        # 1. Write Inputs
        for name, shape in inputs.items():
            f.write(f"INPUT {name} {shape[0]} {shape[1]}\n")
            
        # 2. Write Weights and save as .bin
        for name, arr in initializers.items():
            shape = list(arr.shape)
            if len(shape) == 1:
                shape = (1, shape[0]) # Make bias a row vector to match Add semantics if needed, or col vector.
                # Actually, our C++ Add operates row-by-row or col-by-col? 
                # C++ engine expects Bias as (1, N) or (N, 1). Let's use (1, N).
            elif len(shape) == 2:
                # ONNX Gemm usually has transB=1 (Weight is transposed: out_feats x in_feats)
                # Our C++ engine expects (in_feats x out_feats). So we transpose it.
                arr = arr.T
                shape = arr.shape
            
            # Clean filename by removing special chars
            clean_name = name.replace('/', '_').replace('.', '_')
            f.write(f"INPUT {clean_name} {shape[0]} {shape[1]}\n")
            bin_path = f"{clean_name}.bin"
            arr.astype('float32').tofile(bin_path)
            print(f"Exported weight: {bin_path} {shape}")
            
        # 3. Write Operations
        for node in graph.node:
            clean_out = node.output[0].replace('/', '_').replace('.', '_')
            clean_in0 = node.input[0].replace('/', '_').replace('.', '_')
            
            if node.op_type == "Gemm":
                clean_in1 = node.input[1].replace('/', '_').replace('.', '_')
                
                # Gemm is MatMul + Add
                if len(node.input) > 2:
                    clean_in2 = node.input[2].replace('/', '_').replace('.', '_')
                    matmul_out = f"{clean_out}_matmul"
                    f.write(f"MATMUL {matmul_out} {clean_in0} {clean_in1}\n")
                    f.write(f"ADD {clean_out} {matmul_out} {clean_in2}\n")
                else:
                    f.write(f"MATMUL {clean_out} {clean_in0} {clean_in1}\n")
                    
            elif node.op_type == "MatMul":
                clean_in1 = node.input[1].replace('/', '_').replace('.', '_')
                f.write(f"MATMUL {clean_out} {clean_in0} {clean_in1}\n")
                
            elif node.op_type == "Add":
                clean_in1 = node.input[1].replace('/', '_').replace('.', '_')
                f.write(f"ADD {clean_out} {clean_in0} {clean_in1}\n")
                
            elif node.op_type == "Relu":
                f.write(f"RELU {clean_out} {clean_in0}\n")
                
        # 4. Final Output
        for out in graph.output:
            clean_out = out.name.replace('/', '_').replace('.', '_')
            f.write(f"OUTPUT {clean_out}\n")
            
    print(f"Graph IR exported to {graph_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onnx_exporter.py <model.onnx>")
        sys.exit(1)
    export_onnx(sys.argv[1], "model.graph")
