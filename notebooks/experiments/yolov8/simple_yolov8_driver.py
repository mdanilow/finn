
# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
from qonnx.core.datatype import DataType
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device

from os.path import join
from time import time
import cv2

from yolov8 import yolov8_postproc, make_anchors, plot_one_box

# dictionary describing the I/O of the FINN-generated accelerator
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['UINT8']],
    "odt" : [DataType['INT21'], DataType['INT21'], DataType['INT21']],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute FINN-generated accelerator on numpy inputs, or run throughput test')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="alveo")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--device', help='FPGA device to be used', type=int, default=0)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--inputfile', help='name(s) of input npy file(s) (i.e. "input.npy")', nargs="*", type=str, default=["input.npy"])
    parser.add_argument('--outputfile', help='name(s) of output npy file(s) (i.e. "output.npy")', nargs="*", type=str, default=["output0.npy", "output1.npy", "output2.npy"])
    parser.add_argument('--runtime_weight_dir', help='path to folder containing runtime-writable .dat weights', default="runtime_weights/")
    parser.add_argument('--sequence_dir', help='path to the folder with input images', type=str, default='images')
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    platform = args.platform
    batch_size = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    runtime_weight_dir = args.runtime_weight_dir
    devID = args.device
    device = Device.devices[devID]

    sequence_dir = args.sequence_dir

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir, device=device
    )

    imgnames = os.listdir(sequence_dir)
    imgnames.sort()
    imgnames = imgnames
    muls = [np.load("Mul_{}_param0".format(i)) for i in range(io_shape_dict['num_outputs'])]
    adds = [np.load("Add_{}_param0".format(i)) for i in range(io_shape_dict['num_outputs'])]
    anchor_points, strides = make_anchors(io_shape_dict)

    num_batches = int(np.floor(len(imgnames) / batch_size))
    imgbatches = [imgnames[batch*batch_size : (batch + 1)*batch_size] for batch in range(num_batches)]
    imgbatches = [np.concatenate([np.load(join(sequence_dir, path)) for path in batch], axis=0) for batch in imgbatches]
    outputs = [[np.zeros(1) for _ in range(io_shape_dict['num_outputs'])] for b in range(batch_size)]
    
    start = time()
    for batch_idx, batch in enumerate(imgbatches):

        obuf_normal = accel.execute([batch])
        # out_batches = []
        for o, obuf in enumerate(obuf_normal):
            # np.save(outputfile[o], obuf)
            out = obuf.transpose(0, 3, 1, 2)
            out *= muls[o]
            out += adds[o]
            for in_batch_idx, single_output in enumerate(out):
                outputs[in_batch_idx][o] = single_output
        
        for outs_idx, outs in enumerate(outputs):
            preds = yolov8_postproc(outs, 1, anchor_points, strides)[0]
            for *xyxy, conf, cls in reversed(preds):
                plot_one_box(xyxy, batch[outs_idx], color=(0, 0, 255), line_thickness=1)
            cv2.imwrite('outputs/result{:03d}.jpg'.format(batch_idx*batch_size + outs_idx), batch[outs_idx])
    processing_time = time() - start
    print('fps:', (batch_size * num_batches) / processing_time)