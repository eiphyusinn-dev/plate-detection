import os
from paddle_ocr import predict_and_visualize, recognize_text_in_bboxes, custom_classes,img_path,exp_file
from paddleocr import PaddleOCR
import gdown

def test_ocr_text():
    yolox_s_url = "https://drive.google.com/uc?id=1cLwdSeJ9RVjj3Ok0lkIMF_iN6wAt4EJx"

    weight_path = 'weights/yolox_s.pth'
    if not os.path.exists(weight_path):
            os.makedirs(os.path.dirname(weight_path))
            gdown.download(yolox_s_url, weight_path, quiet=False)
            print('downloaded success')
    else:
        print(f"Weight file already exists at {weight_path}")


    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')
    ckpt_file = weight_path

    # Predict and visualize license plate detection
    outputs, img_info = predict_and_visualize(img_path, exp_file, ckpt_file, custom_classes)

    if outputs[0] is not None:
        bboxes = outputs[0][:, 0:4]
        result_image = img_info["raw_img"]
        ratio = img_info["ratio"]

        ocr_results = recognize_text_in_bboxes(result_image, bboxes, ocr, ratio)

        # Extract detected text
        detected_text = ocr_results[0][1][0][0][1][0] if ocr_results else ""

        # Assert that the detected text is "eps"
        assert detected_text == "PGMN112", f"Detected text was '{detected_text}' instead of 'eps'"
    else:
        assert False, "No objects detected."

        os.remove(weight_path)
        assertFalse(os.path.exists(weight_path))
