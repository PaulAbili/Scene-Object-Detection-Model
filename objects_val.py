from ultralytics import YOLO

model = YOLO("/umbc/class/cmsc475sp26/common/group_5/temp_objects/weights/best.pt")

model.val(
    data="Objects365.yaml",
    project="/umbc/class/cmsc475sp26/common/group_5/val_runs",
    name="val_run_1"
)