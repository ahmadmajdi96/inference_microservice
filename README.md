# Inference Microservice

FastAPI microservice for image inference:
- Accepts a single image or a zip of images
- Runs YOLOv8 segmentation
- Crops detected objects
- Classifies crops using DINO + CLIP features with a logistic regression classifier
- Returns JSON results and serves cropped images

Run
  docker compose up --build

API docs: http://localhost:8001/docs

Endpoints
- Upload single image
  curl -X POST http://localhost:8001/v1/infer/image -F "file=@/path/to/image.jpg"
- Upload zip of images
  curl -X POST http://localhost:8001/v1/infer/zip -F "file=@/path/to/images.zip"
- Upload and reload models
  curl -X POST http://localhost:8001/v1/models/reload \\
    -F "files=@/path/to/logreg_classifier.pkl" \\
    -F "files=@/path/to/label_encoder.pkl" \\
    -F "files=@/path/to/yolo_best.pt"
- Poll job status
  curl http://localhost:8001/v1/jobs/JOB_ID
- Fetch results JSON
  curl http://localhost:8001/v1/jobs/JOB_ID/results
- Fetch a cropped image
  curl http://localhost:8001/v1/jobs/JOB_ID/files/crops/your_crop.jpg -o crop.jpg

Models
Models are loaded on startup from ./models and mounted into the container at /models:
- logreg_classifier.pkl
- label_encoder.pkl
- yolo_best.pt

Notes
- YOLO weights can be swapped by replacing models/yolo_best.pt.
- Jobs output lives under ./jobs/JOB_ID.
- Ensure CLIP/DINO configs match training. Default CLIP model is ViT-B/32 (feature dim 512).
- Annotated images are saved under ./jobs/JOB_ID/annotated and are referenced in results.json.
