import cv2
import numpy as np
import time
from tool import *
## CV argument
FONT_SIZE = 0.5


class SCRFD:
    def __init__(self, model_name, confThreshold=0.5, nmsThreshold=0.5):
        self.inpWidth = 640
        self.inpHeight = 640
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        ## Tensorrt Stuff
        self.scrfd_engine = load_engine(model_name,trt_runtime)
        self.scrfd_content =  self.scrfd_engine.create_execution_context()
        self.scrfd_inputs, self.scrfd_outputs, self.scrfd_bindings,self.scrfd_stream = allocate_buffers(self.scrfd_engine)
        self.eye_engine = load_engine('model/mbv2_trt801.trt',trt_runtime)
        self.eye_content = self.eye_engine.create_execution_context()
        self.eye_inputs, self.eye_outputs, self.eye_bindings,self.eye_stream = allocate_buffers(self.eye_engine)

    def preprocess_image(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(96,96))
        image = (image - image.min())/(image.max() - image.min())
        image = (2 * image) - 1
        return image
    
    def inference_eye(self,img):
        img = self.preprocess_image(img).flatten().astype(np.float16)
        case_num = load_data(pagelocked_buffer=self.eye_inputs[0].host,input_data=img)
        tmp_output = do_inference(self.eye_content, bindings=self.eye_bindings, inputs=self.eye_inputs, outputs=self.eye_outputs, stream=self.eye_stream)
        return np.argmax(tmp_output)

    def inference_face(self,img):
        tmp = []
        case_num = load_data(pagelocked_buffer=self.scrfd_inputs[0].host,input_data=img)
        tmp_output = do_inference(self.scrfd_content, bindings=self.scrfd_bindings, inputs=self.scrfd_inputs, outputs=self.scrfd_outputs, stream=self.scrfd_stream)
        for i in range(3):
            tmp.append(np.expand_dims(tmp_output[i*3].reshape(-1,1),0))
            tmp.append(np.expand_dims(tmp_output[(i*3)+1].reshape(-1,4),0))
            tmp.append(np.expand_dims(tmp_output[(i*3)+2].reshape(-1,10),0))
        return tmp

    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    padw,
                    self.inpWidth - neww - padw,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale) + 1, self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    padh,
                    self.inpHeight - newh - padh,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            img = cv2.resize(
                srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA
            )
        return img, newh, neww, padh, padw

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 128,
            (self.inpWidth, self.inpHeight),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        # Tensorrt need flatten data
        s = time.time()
        outs = self.inference_face(blob.flatten().astype(np.float16))
        e = time.time()
        print(f'Predict face info cost {e-s}')

        s = time.time()
        # inference output
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx * self.fmc][0]
            bbox_preds = outs[idx * self.fmc + 1][0] * stride
            kps_preds = outs[idx * self.fmc + 2][0] * stride
            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
                np.float32
            )
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * self._num_anchors, axis=1
                ).reshape((-1, 2))

            pos_inds = np.where(scores >= self.confThreshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = self.distance2kps(anchor_centers, kps_preds)
            # kpss = kps_preds
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list).ravel()
        # bboxes = np.vstack(bboxes_list) / det_scale
        # kpss = np.vstack(kpss_list) / det_scale
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold
        )
        e = time.time()
        print(f'Inference cost {e-s}')
        for i in indices:
            #i = i[0]
            xmin, ymin, xamx, ymax = (
                int(bboxes[i, 0]),
                int(bboxes[i, 1]),
                int(bboxes[i, 0] + bboxes[i, 2]),
                int(bboxes[i, 1] + bboxes[i, 3]),
            )
            cv2.rectangle(srcimg, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=2)
            # crop eye here
            crop_range = 24
            s = time.time()
            for j in range(2):
                x = int(kpss[i, j, 0])
                y = int(kpss[i, j, 1])
                cv2.rectangle(
                    srcimg,
                    (x - crop_range, y - crop_range),
                    (x + crop_range, y + crop_range),
                    (255, 0, 0),
                    thickness=2,
                )
                # 畫面眼睛左邊先
                eyes_roi = srcimg[
                    y - crop_range : y + crop_range, x - crop_range : x + crop_range
                ]
                res = self.inference_eye(eyes_roi)
                if res == 0 and j == 1:
                        cv2.putText(
                            srcimg,
                            "Left EYES close",
                            (xmin - 200, ymin - 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SIZE,
                            (0, 255, 0),
                            thickness=1,
                        )
                elif res == 0 and j == 0:
                        cv2.putText(
                            srcimg,
                            "Right EYES close",
                            (xmin + 100, ymin - 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SIZE,
                            (0, 255, 0),
                            thickness=1,
                        )

                elif res == 1 and j == 1:
                        cv2.putText(
                            srcimg,
                            "Left EYES open",
                            (xmin - 200, ymin - 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SIZE,
                            (0, 255, 0),
                            thickness=1,
                        )
                else:
                        cv2.putText(
                            srcimg,
                            "Right EYES open",
                            (xmin + 100, ymin - 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SIZE,
                            (0, 255, 0),
                            thickness=1,
                        )
            e = time.time()
            print(f'crop two eyes cost {e-s}')


            # draw key point
            for j in range(5):
                cv2.circle(
                    srcimg,
                    (int(kpss[i, j, 0]), int(kpss[i, j, 1])),
                    1,
                    (0, 255, 0),
                    thickness=-1,
                )
            cv2.putText(
                srcimg,
                str(np.round(scores[i], 3)),
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                thickness=1,
            )
        return srcimg


if __name__ == "__main__":

    # scrfd stuff
    trt_runtime = trt.Runtime(TRT_LOGGER)
    model_path = "model/scrfd_500m_kps_640_640.trt"
    confThreshold, nmsThreshold = 0.5, 0.5
    mynet = SCRFD(model_path, confThreshold=confThreshold, nmsThreshold=nmsThreshold)
    video_capture = cv2.VideoCapture(0)
    # frame setting
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    video_capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    while True:
        _, frame = video_capture.read()
        outimg = mynet.detect(frame)
        cv2.imshow("Video", outimg)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
