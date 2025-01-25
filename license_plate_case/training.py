from ultralytics import YOLO


def train():
    # Escolhi o yoo11m por ter equilibrio entre precisão e velocidade
    model = YOLO("yolo11m.pt")
    model.train(data="data.yml", epochs=100, imgsz=416, batch=6, lr0=1e-4)


if __name__ == "__main__":
    train()
