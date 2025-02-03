
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

from yolov8_utils import plot_one_box, DetectorDriver, scale_coords

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
    parser.add_argument('--sequence_dir', help='path to the folder with input images', type=str, default='img1')
    parser.add_argument('--save_images', help='whether to visualize results by saving images with bounding boxes', action='store_true')
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

    imgnames = os.listdir(args.sequence_dir)
    imgnames.sort()
    num_batches = int(np.floor(len(imgnames) / batch_size))

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir, device=device
    )

    detector_driver = DetectorDriver(
        accel,
        io_shape_dict,
        batch_size=batch_size,
        stride=[8, 16, 32],
        num_classes=80
    )

    prev_batch = None
    batch = None
    imgbatches_names = [imgnames[batch*batch_size : (batch + 1)*batch_size] for batch in range(num_batches)]
    # additional first iter just for preproc, additional last iter just for postproc
    num_iterations = len(imgbatches_names) + 2
    start = time()
    for iteration in range(num_iterations):

        if iteration != 0:
            accel.execute_on_buffers(asynch=True)
            # postproc
            if iteration != 1:
                batch_detections = detector_driver.read_accel_and_postprocess()
                if args.save_images:
                    for outs_idx, detections in enumerate(batch_detections):
                        vis_img = prev_batch[outs_idx]
                        detections[:, :4] = scale_coords(io_shape_dict['ishape_normal'][0][1:3], detections[:, :4], vis_img.shape[:2])
                        for *xyxy, conf, cls in reversed(detections):
                            plot_one_box(xyxy, vis_img, color=(0, 0, 255), line_thickness=1)
                        cv2.imwrite('outputs/class_result{:03d}.jpg'.format((iteration - 2)*batch_size + outs_idx), vis_img)
        
        # preproc
        if args.save_images:
            prev_batch = batch  # save for visualization
        if iteration < num_iterations - 2:
            batch = [cv2.imread(join(args.sequence_dir, path)) for path in imgbatches_names[iteration]]
            detector_driver.preproc_and_write_accel(batch)
            
        if iteration != 0:
            accel.wait_until_finished()
            for o in range(io_shape_dict['num_outputs']):
                accel.copy_output_data_from_device(accel.obuf_packed[o], ind=o)

    processing_time = time() - start
    print('fps:', (batch_size * num_batches) / processing_time)