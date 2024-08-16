import os
import time
import torch
import cv2
import numpy as np
from yolox.exp import get_exp
from yolox.utils import postprocess, fuse_model, get_model_info, vis
from yolox.data.data_augment import ValTransform
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont

# Define your custom classes
custom_classes = ['license_plate']

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=custom_classes,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # Filter bounding boxes based on confidence threshold
            outputs = [out[out[:, 4] >= 0.7] for out in outputs]
            print("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def predict_and_visualize(img_path, exp_file, ckpt_file, class_names, device='cpu', conf_thres=0.7):
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    model.to(device)
    model.eval()

    ckpt = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(model, exp, class_names, device=device)

    outputs, img_info = predictor.inference(img_path)

    return outputs, img_info

def recognize_text_in_bboxes(image, bboxes, ocr, ratio):
    results = []
    for bbox in bboxes:
        # Scale bounding box coordinates back to the original image size
        bbox = bbox / ratio

        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]

        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        if x_min >= x_max or y_min >= y_max:
            continue  # Skip invalid bounding boxes

        cropped_img = image[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:
            continue  # Skip if the crop is empty

        cv2.imwrite("temp_crop.jpg", cropped_img)
        ocr_result = ocr.ocr("temp_crop.jpg", cls=True)
        results.append((bbox, ocr_result))
    return results

def draw_ocr_results(image, results, font_path):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, 20)

    for bbox, ocr_result in results:
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
        
        if ocr_result is not None:
            for res in ocr_result:
                if res is not None:
                    for line in res:
                        txt = line[1][0]
                        draw.text((x_min, y_min - 25), txt, fill="green", font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



img_path = "../datasets/Cars1.png"   # replace with your image path
exp_file = "../exps/example/custom/yolox_s.py"  # replace with your experiment file path
ckpt_file = "../YOLOX_outputs/yolox_s/best_ckpt.pth"  # replace with your checkpoint path
class_names = custom_classes

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')

# Predict and visualize license plate detection
outputs, img_info = predict_and_visualize(img_path, exp_file, ckpt_file, class_names)

if outputs[0] is None:
    print("No objects detected.")
else:
    # Extract bounding boxes
    bboxes = outputs[0][:, 0:4]

    # Print detected bounding boxes
    print(f"Detected bounding boxes: {bboxes}")

    # Recognize text in bounding boxes
    result_image = img_info["raw_img"]
    ratio = img_info["ratio"]
    ocr_results = recognize_text_in_bboxes(result_image, bboxes, ocr, ratio)

    # Print OCR results
    print(f"OCR results: {ocr_results}")

    # Draw OCR resultsr_img/
    font_path = 'fonts/french.ttf'
    final_image = draw_ocr_results(result_image, ocr_results, font_path)

    # Display the final image with bounding boxes and recognized text
    # cv2.imshow("Final Result", final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the final result image in the current script folder
    # final_result_path = "result.jpg"
    # cv2.imwrite(final_result_path, final_image)
    # print(f"Final result saved to {final_result_path}")
