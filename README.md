# SCRFD_and_eye_closed_tensorrt

* A solution to get 5-face key points and make eye closed detect on jetson nano with tensorrt

## Check your System

    ## Same as jetpack4.6
    tensort version == 8.0.1
    CUDA == 10.2
    CUDNN == 8.2.1
    ##Python package
    pycuda
    tensorrt
 
## SCRFD Model and Reference
[SCRFD](https://insightface.ai/scrfd) and [A good example if you want to use SCRFD with opencv in python or c++](https://github.com/hpc203/scrfd-opencv)

## Eye closed model
Train with [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) with **Mobilenet_V2**

### Result on Test Data (6835 picture)
| ACC| PRE | RECALL | F1_SCORE | FP | FN |
| -------- | -------- | -------- | -------- | -------- | -------- |
|0.97 | 0.99   | 0.95| 0.97    | 27    | 213     |

## Performance on Jeston Nano Under 640 X 640
| Crop face | NMS & Achor | Crop two  Eyes + Predict | Total |
| -------- | -------- | -------- | -------- | 
|0.054 sec | 0.01 sec  | 0.019 sec|  0.083 sec    |

## Export A model

    from tool import *
    engin = build_engine(onnx_path, shape) #EX: [1,96,96,3].
    save_engine(engine, file_name)
        
## Make Sure Your mmcv-full

    pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
