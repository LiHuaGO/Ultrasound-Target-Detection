from ultralytics import YOLO,RTDETR
import torch
# torch.use_deterministic_algorithms(False)  # 关闭确定性计算
# torch.backends.cudnn.deterministic = False  # 关闭 cuDNN 确定性
if __name__ == '__main__':

    yaml = r"rtdetr-n.yaml"
    data = r"/data/lihua/超声/中文期刊_部分数据/YOLO13/data80%_11.19/dataset.yaml"
    project = "腹部"
    name = "rtdetr-n"
    path = project + "/" + name
    model = RTDETR(yaml)#YOLO(yaml, task="detect")
    model.train(data=data, project=project, name=name, imgsz=640, batch=128,patience=False,
                    epochs=300, device=[0,1,2,3,4,5,6,7], exist_ok=True,cos_lr=True,pretrained=True)

    # model = YOLO(path + "/weights/best.pt")
    # model.val(data=data, batch=1,project=path, name="test", split="test", plots=True, exist_ok=True, device=0)