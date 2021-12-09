import tensorrt as trt
import pycuda.driver as cuda

# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def build_engine(onnx_path, shape):
    with trt.Builder(
        TRT_LOGGER
    ) as builder, builder.create_builder_config() as config, builder.create_network(
        explicit_batch
    ) as network, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser:
        if builder.platform_has_fast_fp16:
            print("Use FP 16")
            #builder.fp16_mode = True
        config.max_workspace_size = 1 << 20
        print("parsing")
        with open(onnx_path, "rb") as model:
            print("onnx found")
            if not parser.parse(model.read()):
                print("parse failed")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            # parser.parse(model.read())
        last_layer = network.get_layer(network.num_layers - 1)
        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))
        network.get_input(0).shape = shape
        plan = builder.build_serialized_network(network,config)
        # engine = builder.build_cuda_engine(network)
        return trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(plan)


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, "wb") as f:
        f.write(buf)


def load_engine(engine_path, trt_runtime):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def load_data(pagelocked_buffer, input_data):
    return np.copyto(pagelocked_buffer, input_data)
