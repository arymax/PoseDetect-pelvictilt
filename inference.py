import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import os
import imghdr
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
def draw_and_annotate_body_parts_area(img, midpoint):
    # 使用Canny进行边缘检测
    edges = cv2.Canny(img, 100, 200)

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    front_area = 0
    back_area = 0

    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

         # 根据midpoint的x坐标，将轮廓分为前后两部分并绘制
        if x + w/2 < midpoint[0]:
            back_area += cv2.contourArea(contour)
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)  # 使用红色绘制后部分
        else:
            front_area += cv2.contourArea(contour)
            cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)  # 使用绿色绘制前部分

    # 在图像上标注每个部分的面积
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Front area: {front_area:.2f}", (10, 30), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Back area: {back_area:.2f}", (10, 60), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return img
def initialize_mmdet_model(config_file, checkpoint_file, device='cpu'):
    """初始化mmdet模型并返回"""
    model = init_detector(config_file, checkpoint_file, device=device)
    return model

def get_bbox_from_mmdet(model, image_path):
    """使用mmdet模型获取bbox"""
    results = inference_detector(model, image_path)
    bbox = results.pred_instances.bboxes.tolist()
    # 返回最大的bbox
    sorted_bboxes = sorted(bbox, key=lambda bbox: (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]), reverse=True)
    if sorted_bboxes:
        return sorted_bboxes[0]
    return None

def crop_image_using_bbox(image_path, bbox):
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = img[y1:y2, x1:x2]
    temp_file_path = "temp_cropped_image.jpg"
    cv2.imwrite(temp_file_path, cropped_img)
    return temp_file_path
def get_model_files(checkpoint_dir):
    pattern = re.compile(r'best_coco_AP_epoch_\d+.pth')
    return [f for f in os.listdir(checkpoint_dir) if pattern.match(f)]

def get_image_file(image_file):
    file_extension = os.path.splitext(image_file)[1]
    if file_extension.lower() != '.jpg':
        img = cv2.imread(image_file)
        new_image_file = os.path.splitext(image_file)[0] + '.jpg'
        return new_image_file
    return image_file

def get_keypoints_bbox_and_ids(model, image_file, config_file, checkpoint_file):    
    results = inference_topdown(model, image_file)
    keypoints = results[0].pred_instances.keypoints[0].tolist()   
    keypoint_id2name = {0: 'head', 1: 'shoulder', 2: 'psis', 3: 'asis', 4: 'knee', 5: 'ear', 6: 'midpoint_psis_asis'}
    return keypoints, keypoint_id2name   

def extract_coordinates_from_keypoints(keypoints, keypoint_id2name):
    coords = {'psis': np.array([0, 0]), 'asis': np.array([0, 0]), 'knee': np.array([0, 0]), 'midpoint_psis_asis': np.array([0, 0])}
    for i, (x, y) in enumerate(keypoints):
        if keypoint_id2name[i] in coords:
            coords[keypoint_id2name[i]] = np.array([x, y])
            print(f"Index: {i}, Name: {keypoint_id2name[i]}, Coordinates: ({x}, {y})")
    return coords

def calculate_angle_between_vectors(vector1, vector2):
    angle = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])
    return np.degrees(angle)

def adjust_angle(angle):
    return angle - 180 if angle > 180 else angle

def visualize_vectors_on_image(img, coords):
    cv2.arrowedLine(img, tuple(coords['psis'].astype(int)), tuple(coords['asis'].astype(int)), (0, 255, 0), 3)
    cv2.arrowedLine(img, tuple(coords['midpoint_psis_asis'].astype(int)), tuple(coords['knee'].astype(int)), (0, 0, 255), 3)
    return img
def is_image_file(filepath):
    """检查文件是否为图像"""
    return imghdr.what(filepath) is not None

register_all_modules()
# 初始化mmdet模型
config_file_mmdet = 'mask-rcnn_r50-caffe_fpn_1x_coco.py'
checkpoint_file_mmdet = 'mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'
model_mmdet = initialize_mmdet_model(config_file_mmdet, checkpoint_file_mmdet)

config_file = '/data1/data/vscode-ml/posedetect/pelvictilt-config.py'
checkpoint_dir = '/data1/data/vscode-ml/posedetect/checkpoint/'
pattern = re.compile(r'best_coco_AP_epoch_\d+.pth')
#model_files = [f for f in os.listdir(checkpoint_dir) if pattern.match(f)]
model_files = ['best_coco_AP_epoch_70.pth']

INPUT_IMAGE_DIR = './inference'  

# 获取文件夹中的所有文件
all_files = [os.path.join(INPUT_IMAGE_DIR, f) for f in os.listdir(INPUT_IMAGE_DIR)]
image_files = [f for f in all_files if is_image_file(f)]
""", '22836_0.jpg','22841_0.jpg','IMG_3089.jpg'"""
def process_image(image_path):
    # 获取bbox并裁剪图像
    bbox = get_bbox_from_mmdet(model_mmdet, image_path)
    cropped_image_path = crop_image_using_bbox(image_path, bbox)
    
    for idx_model, model_file in enumerate(model_files):
        checkpoint_file = os.path.join(checkpoint_dir, model_file)
        model = init_model(config_file, checkpoint_file, device='cuda:0')
        keypoints, keypoint_id2name = get_keypoints_bbox_and_ids(model, cropped_image_path, config_file, checkpoint_file)
        coords = extract_coordinates_from_keypoints(keypoints, keypoint_id2name)
        vector1 = coords['asis'] - coords['psis']
        vector2 = coords['knee'] - coords['midpoint_psis_asis']
        angle = calculate_angle_between_vectors(vector1, vector2)
        angle = adjust_angle(angle)
        img = cv2.imread(cropped_image_path)
        midpoint = coords['midpoint_psis_asis']    
         # 在图像上绘制并计算左右两侧的面积
        
        img = visualize_vectors_on_image(img, coords)
       # 获取图像尺寸
        height, width, channels = img.shape       


        # 动态设置弧线半径和字体大小
        radius = int(width * 0.05)  # 假设半径是图像宽度的5%
        font_scale = width / 600  # 假设字体大小与图像宽度成正比
        # 绘制从coords['midpoint_psis_asis']正上方的垂直线
        line_length = int(height * 0.1)  # 假设线的长度是图像高度的10%
        start_point = (int(coords['midpoint_psis_asis'][0]), int(coords['midpoint_psis_asis'][1]))
        end_point = (int(coords['midpoint_psis_asis'][0]),0)
        cv2.line(img, start_point, end_point, (0, 255, 255), 3)  # 使用黄色绘制线条
        # 计算弧线的开始和结束角度 
        angle_start = np.degrees(np.arctan2(vector1[1], vector1[0]))
        angle_end = angle_start + angle
        print(angle)
        img = draw_and_annotate_body_parts_area(img, coords['midpoint_psis_asis'])
        # 在图像上绘制角度的弧
        cv2.ellipse(img, tuple(coords['midpoint_psis_asis'].astype(int)), (radius, radius), 0, angle_start, angle_end, (255, 0, 0), 2)
        # 在图像上标注角度
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'Angle: {abs(angle):.2f}', (int(coords['midpoint_psis_asis'][0]), int(coords['midpoint_psis_asis'][1] - radius - 10)), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        output_file = f"output_{model_file.split('.')[0]}.jpg"
        cv2.imwrite(output_file, img)
    
    os.remove(cropped_image_path)
    return output_file
