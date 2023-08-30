import os
import io
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model_alexnet import AlexNet

app = Flask(__name__)
CORS(app)  # 解决跨域问题

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
model = AlexNet(num_classes=5).to(device)
# ort_session = onnxruntime.InferenceSession(r'E:\\模型部署\\7-ONNX Runtime图像分类部署\\1-Pytorch图像分类模型转ONNX\\Alex_flower5.onnx')
def load_onnx_model():
    global ort_session
    ort_session = onnxruntime.InferenceSession(r'E:\\模型部署\\7-ONNX Runtime图像分类部署\\1-Pytorch图像分类模型转ONNX\\Alex_flower5.onnx')

# 在应用启动时加载 ONNX 模型
load_onnx_model()

model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(227),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    # 记录该帧开始处理的时间
    start_time = time.time()
    tensor = transform_image(image_bytes=image_bytes)

    ort_inputs = {'input': tensor.numpy()}
    pred_logits = ort_session.run(['output'], ort_inputs)[0]
    pred_logits = torch.from_numpy(pred_logits)

    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
    top_n = torch.topk(pred_softmax, 5)

    pred_ids = top_n.indices[0].cpu().detach().numpy()  # 将索引转换为NumPy数组，并分离梯度
    confs = top_n.values[0].cpu().detach().numpy() * 100  # 将置信度转换为NumPy数组，并分离梯度

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)

    # 载入类别和对应 ID
    idx_to_labels = np.load('idx_to_labels1.npy', allow_pickle=True).item()

    results = []  # 用于存储结果的列表
    for i in range(5):
        class_name = idx_to_labels[pred_ids[i]]  # 获取类别名称
        confidence = confs[i]  # 获取置信度
        text = '{:<6} {:>.3f}'.format(class_name, confidence)
        results.append(text)  # 将结果添加到列表中


    return results, FPS  # 返回包含类别和置信度的列表


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_info, FPS = get_prediction(image_bytes=img_bytes)
        response_data = {'class_info': class_info, 'FPS': FPS}
        return jsonify(response_data)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up_my.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




