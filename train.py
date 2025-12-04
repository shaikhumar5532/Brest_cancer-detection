from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"C:\Users\AbdulBashar\Downloads\Breast Cancer Detection.v2i.yolov8\Dataset\data.yaml",
        epochs=40,
        device=0,
        workers=8,
        imgsz=416,
        batch=16,
        project="YOLOv8",
        name="bc_detection",
        resume=False,       # resume=True caused issues earlier
        pretrained=True     # keep pretrained weights
    )

    print("🔥 Training started on GPU successfully!")

if __name__ == "__main__":
    main()
