import numpy as np


STRIDE = [8, 16, 32]
io_shape_dict = {
    # FINN DataType for input and output tensors
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 192, 320, 3)],
    "oshape_normal" : [(1, 24, 40, 144), (1, 12, 20, 144), (1, 6, 10, 144)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 192, 320, 3, 1)],
    "oshape_folded" : [(1, 24, 40, 144, 1), (1, 12, 20, 144, 1), (1, 6, 10, 144, 1)],
    "ishape_packed" : [(1, 192, 320, 3, 1)],
    "oshape_packed" : [(1, 24, 40, 144, 3), (1, 12, 20, 144, 3), (1, 6, 10, 144, 3)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0', 'odma1', 'odma2'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 3,
}

def make_anchors(io_shape_dict, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    output_shapes = io_shape_dict["oshape_normal"]
    for i, stride in enumerate(strides):
        _, h, w, _ = output_shapes[i]
        sx = np.arange(start=grid_cell_offset, stop=w, step=1)
        sy = np.arange(start=grid_cell_offset, stop=h, step=1)
        sx, sy = np.meshgrid(sx, sy)
        anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
        stride_tensor.append([stride] * (h*w))

    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


points, strides = make_anchors(io_shape_dict, STRIDE)

print(points.shape)
print(strides.shape)
print(points)
print(strides)
