import cv2
import numpy as np
import onnxruntime as ort


palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
radius = 5


class YOLOV7POSEONNX(object):
    def __init__(
        self,
        model_path,
        input_shape,
        score_thr,
        cuda = None,
    ):
        self.input_shape = input_shape
        self.score_thr = score_thr

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
    
    def inference(self, img_origin):
        img = img_origin[:, :, ::-1]
        input, ratio, dwdh = self.preproc(img)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run([], {input_name: input})[0]

        reslut_img = self.post_process(img_origin, output, ratio, dwdh, self.score_thr)

        return reslut_img
    
    def preproc(self, img, img_mean=0.0, img_scale=0.00392156862745098): #img_mean=127.5, img_scale=1/127.5):
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        return im, ratio, dwdh
    
    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape=(self.input_shape, self.input_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)
    
    def post_process(self, img_origin, output, ratio, dwdh, score_threshold=0.3):
        det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
        
        for idx in range(len(det_bboxes)):
            kpt = kpts[idx]
            if det_scores[idx]>score_threshold:
                img_origin = self.plot_skeleton_kpts(img_origin, kpt, ratio, dwdh)
        
        return img_origin
    
    def plot_skeleton_kpts(self, im, kpts, ratio, dwdh, steps=3):
        num_kpts = len(kpts) // steps
        #plot keypoints
        for kid in range(num_kpts):
            r, g, b = pose_kpt_color[kid]
            coord = np.array([kpts[steps * kid], kpts[steps * kid + 1]])
            coord -= np.array(list(dwdh))
            coord /= ratio
            coord  = coord .round().astype(np.int32).tolist()
            x_coord, y_coord = coord[0], coord[1]
            conf = kpts[steps * kid + 2]
            if conf > 0.5: #Confidence of a keypoint has to be greater than 0.5
                cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
        #plot skeleton
        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            posi = np.array([kpts[(sk[0]-1)*steps], kpts[(sk[0]-1)*steps+1], kpts[(sk[1]-1)*steps], kpts[(sk[1]-1)*steps+1]])
            posi -= np.array(dwdh*2)
            posi /= ratio
            posi = posi.round().astype(np.int32).tolist()
            pos1 = (int(posi[0]), int(posi[1]))
            pos2 = (int(posi[2]), int(posi[3]))
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1>0.5 and conf2>0.5: # For a limb, both the keypoint confidence must be greater than 0.5
                cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        return im