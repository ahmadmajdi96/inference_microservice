from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "inference-microservice"
    models_dir: str = "/models"
    jobs_dir: str = "/jobs"

    yolov8_weights: str = "/models/yolo_best.pt"
    logreg_path: str = "/models/logreg_classifier.pkl"
    label_encoder_path: str = "/models/label_encoder.pkl"

    dino_model: str = "vit_small_patch14_dinov2"
    clip_model: str = "ViT-B/32"

    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    max_det: int = 300

    device: str | None = None

    shelf_gap_ratio: float = 0.6
    shelf_gap_min_px: int = 40

settings = Settings()
