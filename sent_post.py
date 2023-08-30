import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('flower.jpg','rb')})

if resp.status_code == 200:
    response_data = resp.json()
    class_info = response_data['class_info']
    fps = response_data['FPS']
    # 输出类别信息
    for info in class_info:
        print(info)
    # print("Class Info:", class_info)
    print("FPS:", fps)
else:
    print("Error:", resp.text)

