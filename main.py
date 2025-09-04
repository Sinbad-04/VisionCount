from ultralytics import YOLO

def main():

    model = YOLO(r"runs/detect/train8/weights/last.pt")

    model.train(data="data.yaml", epochs=100, imgsz=640,device = 0, batch = 16, augment = True)


if __name__ == '__main__':
    main()
    #74