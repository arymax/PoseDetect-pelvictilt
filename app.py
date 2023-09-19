from flask import Flask, request, send_from_directory
import os
from inference import process_image
app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    # 保存上传的图像
    image = request.files['image']
    image_path = os.path.join("./inference", image.filename)
    print(image_path)
    image.save(image_path)

    # 调用你的推理函数
    output_path = process_image(image_path)

    # 返回处理后的图像
    
    return "Image processed successfully!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
