from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="Objects365.yaml", 
    epochs=100, 
    imgsz=640, 
    device=0,
    batch=16,
    optimizer="AdamW",
    weight_decay=0.0005,
    lr0=1e-3,
    project="runs",
    name="trained_objects",
    save=True
    )