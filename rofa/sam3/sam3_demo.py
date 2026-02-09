import requests
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="ear")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# 保存结果
import numpy as np
import cv2
masks = masks.cpu().numpy()
boxes = boxes.cpu().numpy()
scores = scores.cpu().numpy()
print("Masks shape:", masks.shape)
print("Boxes shape:", boxes.shape)
print("Scores shape:", scores.shape)
# 可视化结果
for i in range(len(masks)):
    mask = masks[i]
    box = boxes[i]
    score = scores[i]
    print(f"Mask {i} shape: {mask.shape}, Box: {box}, Score: {score}")
    # 可视化掩码
    mask_vis = (mask.squeeze(0) * 255).astype(np.uint8)  # (1, H, W) -> (H, W)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    # 可视化边界框
    image_np = np.array(image)[..., ::-1]  # RGB -> BGR for OpenCV
    box_vis = image_np.copy()
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(box_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 显示结果
    combined_vis = cv2.addWeighted(mask_vis, 0.5, image_np, 0.5, 0)
    combined_vis = cv2.rectangle(combined_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(f"result_{i}.png", combined_vis)
