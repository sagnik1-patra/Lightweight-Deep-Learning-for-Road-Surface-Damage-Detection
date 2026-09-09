# Lightweight Deep Learning for Road Surface Damage Detection

## Overview

Road surface damage such as potholes can affect vehicle safety, traffic flow, and road maintenance costs. Manual road inspection is time-consuming and difficult to scale across large road networks.

This project presents a **lightweight deep learning-based pothole detection system** using **YOLOv8n**, a compact object detection model designed for efficient inference.

The system detects potholes from road images, draws bounding boxes around damaged regions, generates confidence scores, evaluates model performance, and saves detailed prediction results and visualizations.

The project focuses not only on detection accuracy but also on **model size, inference speed, and computational efficiency**, making it suitable for possible deployment on edge devices, mobile systems, drones, or road-monitoring platforms.

---

# Research Title

**Lightweight Deep Learning for Road Surface Damage Detection Using YOLOv8n**

---

# Research Objective

The main objective of this project is to develop a lightweight computer vision model capable of automatically detecting potholes from road images.

The research evaluates whether a compact object detection model such as **YOLOv8n** can provide reliable pothole detection while maintaining low computational requirements.

The project evaluates the system using:

- Precision
- Recall
- mAP@0.50
- mAP@0.50:0.95
- Model size
- Inference time
- Frames Per Second (FPS)
- Prediction confidence
- Bounding box characteristics
- Pothole count error

---

# Problem Statement

Traditional road inspection methods usually depend on manual observation or physical inspection. These approaches can be expensive, slow, and difficult to perform continuously.

Computer vision provides an automated solution by detecting damaged road regions directly from images.

However, many deep learning object detection models require high computational resources.

Therefore, this project investigates the use of a **lightweight YOLOv8 Nano model** to detect potholes while maintaining a balance between:

- Detection accuracy
- Computational efficiency
- Model size
- Inference speed

---

# Dataset

The project uses a pothole detection dataset containing approximately:

| Dataset Split | Number of Images |
|---|---:|
| Training | 465 |
| Validation | 133 |
| Testing | 67 |
| **Total** | **665** |

The dataset contains annotated road images with bounding boxes identifying potholes.

The object detection class used in this project is:

```text
0 - pothole
```

The annotations follow the **YOLO object detection format**.

Each annotation contains:

```text
class_id x_center y_center width height
```

where all coordinates are normalized relative to the image dimensions.

---

# Dataset Structure

The dataset is organized as:

```text
archive/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

---

# Proposed Methodology

The complete workflow of the proposed system is:

```text
Road Image Dataset
        |
        v
Dataset Inspection
        |
        v
Image and Label Verification
        |
        v
Data Augmentation
        |
        v
YOLOv8n Model
        |
        v
Model Training
        |
        v
Validation
        |
        v
Test Evaluation
        |
        v
Pothole Detection
        |
        v
Bounding Box + Confidence Score
        |
        v
Performance Analysis
        |
        v
CSV + Graph + Model Output
```

---

# Why YOLOv8n?

YOLOv8n is the **Nano version of YOLOv8**.

It was selected because the main objective of the project is lightweight object detection.

Compared with larger YOLO models, YOLOv8n provides:

- Smaller model size
- Lower memory requirements
- Faster inference
- Reduced computational cost
- Good object detection performance
- Suitability for real-time applications

These characteristics make it suitable for applications such as:

- Smartphone-based road inspection
- Drone-based road monitoring
- Vehicle-mounted cameras
- Edge computing devices
- Smart city monitoring systems

---

# Model Architecture

The general detection process is:

```text
Input Road Image
        |
        v
Image Resizing
        |
        v
YOLOv8n Backbone
        |
        v
Feature Extraction
        |
        v
Feature Fusion
        |
        v
Detection Head
        |
        v
Bounding Box Prediction
        |
        v
Confidence Score
        |
        v
Pothole Detection
```

YOLOv8 performs object localization and classification simultaneously.

For every detected pothole, the system returns:

- Object class
- Confidence score
- X1 coordinate
- Y1 coordinate
- X2 coordinate
- Y2 coordinate
- Bounding box width
- Bounding box height
- Bounding box area

---

# Image Preprocessing

Images are automatically resized during YOLO training.

The input image size used in this project is:

```text
640 × 640
```

The model performs preprocessing internally before passing images through the neural network.

---

# Data Augmentation

Data augmentation is used to improve generalization and reduce overfitting.

The training configuration includes augmentation operations such as:

```text
Horizontal Flip
Translation
Scaling
Small Rotation
Mosaic Augmentation
```

Vertical flipping is disabled because vertically inverted road scenes are generally unrealistic.

Example training parameters:

```python
degrees=5
translate=0.1
scale=0.4
fliplr=0.5
flipud=0.0
mosaic=1.0
```

---

# Training Configuration

The primary configuration used for the experiment is:

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Input Resolution | 640 × 640 |
| Epochs | 50 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.001 |
| Early Stopping Patience | 15 |
| Random Seed | 42 |

The pre-trained YOLOv8n weights are used as the initial model.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

This uses **transfer learning**, allowing the model to benefit from previously learned visual features.

---

# Transfer Learning

Training a deep neural network completely from scratch generally requires a very large dataset.

Because the pothole dataset contains only several hundred images, this project uses transfer learning.

YOLOv8n is initialized using pre-trained weights and then fine-tuned on the pothole dataset.

Transfer learning provides several advantages:

- Faster convergence
- Better performance on smaller datasets
- Reduced training requirements
- Improved feature extraction

---

# Evaluation Metrics

The model is evaluated using multiple object detection metrics.

## Precision

Precision measures the proportion of detected potholes that are actually correct.

```text
Precision = TP / (TP + FP)
```

A high precision means the model produces fewer false pothole detections.

## Recall

Recall measures the proportion of actual potholes that the model successfully detects.

```text
Recall = TP / (TP + FN)
```

High recall indicates that the model misses fewer potholes.

## mAP@0.50

Mean Average Precision at IoU 0.50 measures overall object detection performance using an Intersection over Union threshold of 0.50.

Higher values indicate better detection performance.

## mAP@0.50:0.95

This metric evaluates detection across multiple IoU thresholds ranging from:

```text
0.50 to 0.95
```

It provides a stricter and more comprehensive evaluation of detection quality.

---

# Prediction Confidence

Every pothole detection receives a confidence value between:

```text
0 and 1
```

For example:

```text
Pothole Confidence = 0.91
```

means that the model has approximately 91% confidence in the detection.

A confidence threshold of:

```text
0.25
```

is used during prediction.

---

# Main Visualization

The following graph analyzes the relationship between the predicted bounding box area and the confidence of each pothole detection.

![Confidence vs Bounding Box Area](road_damage_results/graphs/confidence_vs_bbox_area.png)

### Figure: Prediction Confidence vs Bounding Box Area

This visualization helps evaluate whether the physical size of a pothole region in the image influences the confidence of the model.

The **X-axis** represents:

```text
Predicted Bounding Box Area
```

The **Y-axis** represents:

```text
Prediction Confidence
```

Each point represents an individual pothole detected by the trained model.

Large bounding boxes typically represent potholes occupying a larger portion of the image, while smaller bounding boxes may represent distant or smaller potholes.

This analysis is useful because smaller road defects may be more difficult for the object detection model to identify confidently.

---

# Additional Visualizations

The project generates several graphs automatically:

```text
dataset_distribution.png
bounding_box_distribution.png
bounding_box_area_distribution.png
training_loss_graph.png
validation_loss_graph.png
precision_recall_training_graph.png
map_training_graph.png
confusion_matrix.png
confusion_matrix_normalized.png
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
validation_test_metric_comparison.png
test_prediction_samples.png
prediction_confidence_distribution.png
detected_potholes_per_image.png
confidence_vs_bbox_area.png
ground_truth_vs_prediction_graph.png
inference_speed_distribution.png
final_performance_graph.png
```

---

# Training Loss Analysis

The model records multiple loss components during training.

## Box Loss

Measures the error in predicting pothole bounding box coordinates.

Lower values indicate more accurate object localization.

## Classification Loss

Measures the error associated with classification predictions.

Since this dataset contains one class, the model primarily learns to distinguish potholes from the background.

## Distribution Focal Loss

Distribution Focal Loss helps improve bounding box localization accuracy.

The loss values should generally decrease as training progresses.

---

# Prediction Results

The trained model generates predictions for every test image.

Each detected pothole includes:

```text
Image Name
Detection ID
Class ID
Class Name
Confidence
X1
Y1
X2
Y2
Bounding Box Width
Bounding Box Height
Bounding Box Area
Image Width
Image Height
```

These results are stored in:

```text
detection_predictions.csv
```

---

# Image-Level Prediction Results

The project also generates an image-level summary.

For every image, the following information is stored:

```text
image
pothole_detected
number_of_potholes
maximum_confidence
mean_confidence
```

The results are stored in:

```text
image_level_predictions.csv
```

---

# Ground Truth vs Prediction Analysis

The number of actual potholes in every test image is compared with the number predicted by the model.

The comparison is stored in:

```text
ground_truth_vs_prediction_counts.csv
```

The difference between actual and predicted counts is calculated as:

```text
Absolute Error =
|Ground Truth Potholes - Predicted Potholes|
```

The average of these errors provides the:

```text
Mean Absolute Pothole Count Error
```

---

# Inference Performance

Since the project focuses on lightweight deep learning, inference speed is also evaluated.

For every test image, the system records:

```text
Inference Time in Milliseconds
```

The average inference speed is then calculated.

Approximate Frames Per Second is calculated as:

```text
FPS = 1000 / Average Inference Time in ms
```

Higher FPS indicates faster model performance.

> **Note:** Inference speed depends on the hardware, operating system, PyTorch configuration, and whether CPU or GPU acceleration is used.

---

# Lightweight Model Evaluation

The project evaluates both detection performance and computational efficiency.

The primary lightweight characteristics include:

```text
Model File Size
Inference Time
Frames Per Second
Detection Performance
```

This allows the system to be evaluated not only as a pothole detector but also as a candidate for potential real-time deployment.

---

# Output Directory

All generated files are stored inside:

```text
C:\Users\sagni\Downloads\Lightweight Deep Learning for Road Surface Damage Detection\road_damage_results
```

The output structure is:

```text
road_damage_results/
│
├── models/
│   ├── road_damage_yolov8n_best.pt
│   └── road_damage_yolov8n_last.pt
│
├── csv/
│   ├── dataset_distribution.csv
│   ├── bounding_box_distribution.csv
│   ├── bounding_box_analysis.csv
│   ├── training_results.csv
│   ├── validation_metrics.csv
│   ├── test_metrics.csv
│   ├── validation_test_comparison.csv
│   ├── detection_predictions.csv
│   ├── image_level_predictions.csv
│   ├── ground_truth_vs_prediction_counts.csv
│   ├── inference_speed.csv
│   └── final_research_results.csv
│
├── graphs/
│   ├── dataset_distribution.png
│   ├── bounding_box_distribution.png
│   ├── bounding_box_area_distribution.png
│   ├── training_loss_graph.png
│   ├── validation_loss_graph.png
│   ├── precision_recall_training_graph.png
│   ├── map_training_graph.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── PR_curve.png
│   ├── F1_curve.png
│   ├── P_curve.png
│   ├── R_curve.png
│   ├── validation_test_metric_comparison.png
│   ├── test_prediction_samples.png
│   ├── prediction_confidence_distribution.png
│   ├── detected_potholes_per_image.png
│   ├── confidence_vs_bbox_area.png
│   ├── ground_truth_vs_prediction_graph.png
│   ├── inference_speed_distribution.png
│   └── final_performance_graph.png
│
├── predictions/
│   └── annotated_images/
│
├── config/
│   └── pothole_dataset.yaml
│
├── final_research_results.json
├── final_research_results.yaml
└── research_summary.txt
```

---

# Requirements

The project requires **Python 3.x**.

Install the required libraries using:

```bash
pip install ultralytics pandas numpy matplotlib scikit-learn pyyaml opencv-python tqdm
```

Main libraries used:

- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyYAML
- TQDM

---

# Running the Project

## Step 1 — Install Dependencies

```bash
pip install ultralytics pandas numpy matplotlib scikit-learn pyyaml opencv-python tqdm
```

## Step 2 — Add Dataset

Place the dataset inside:

```text
C:\Users\sagni\Downloads\Lightweight Deep Learning for Road Surface Damage Detection\archive
```

## Step 3 — Verify Dataset Structure

Make sure the following directories exist:

```text
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
```

## Step 4 — Start Jupyter Notebook

```bash
jupyter notebook
```

## Step 5 — Run Dataset Analysis

Run the notebook cells sequentially to:

- Check dataset directories
- Count images
- Count annotations
- Analyze bounding boxes
- Generate dataset graphs

## Step 6 — Train the Model

The model automatically downloads:

```text
yolov8n.pt
```

if it is not already available.

## Step 7 — Evaluate the Model

Evaluate the best trained model using the validation and test sets.

## Step 8 — Generate Predictions

Generate predictions for all test images.

## Step 9 — Generate Research Graphs

All research graphs are saved under:

```text
road_damage_results/graphs/
```

## Step 10 — Review Final Results

Final summarized results are available in:

```text
final_research_results.csv
final_research_results.json
final_research_results.yaml
research_summary.txt
```

---

# Example Prediction

The final model produces predictions conceptually similar to:

```text
Image: pothole_test_01.jpg

Detected Object:
Class       : Pothole
Confidence  : 0.92

Bounding Box:
X1          : 185
Y1          : 240
X2          : 471
Y2          : 411
```

The bounding box is drawn directly over the detected road damage region.

---

# Saved Models

Two primary model files are generated.

## Best Model

```text
road_damage_yolov8n_best.pt
```

This corresponds to the best-performing checkpoint observed during training.

## Last Model

```text
road_damage_yolov8n_last.pt
```

This contains the weights from the final training epoch.

For inference and deployment, the recommended model is:

```text
road_damage_yolov8n_best.pt
```

---

# Research Contribution

The main contribution of this project is the development and evaluation of a **lightweight pothole detection framework using YOLOv8n**.

The framework investigates the trade-off between:

```text
Detection Performance
        VS
Computational Efficiency
```

Instead of evaluating only detection performance, the research also considers:

- Model size
- Inference latency
- FPS
- Bounding box confidence
- Object count accuracy
- Detection confidence relative to pothole size

This makes the project more suitable for practical real-time applications.

---

# Possible IEEE Paper Contributions

The work can be presented through the following contributions:

1. Development of a lightweight YOLOv8n-based pothole detection framework.
2. Evaluation of pothole detection using Precision, Recall, mAP@0.50, and mAP@0.50:0.95.
3. Analysis of model efficiency through model size, inference latency, and FPS.
4. Investigation of the relationship between pothole bounding box size and model confidence.
5. Comparison between ground-truth pothole counts and model predictions.
6. Generation of a reproducible pipeline for road surface damage analysis.

---

# Research Questions

## RQ1

**Can YOLOv8n provide reliable pothole detection performance using a relatively small road damage dataset?**

## RQ2

**Can a lightweight object detection architecture provide sufficient inference speed for possible real-time road monitoring?**

## RQ3

**How does pothole bounding box size affect prediction confidence?**

## RQ4

**How accurately can the model estimate the number of potholes present in test images?**

---

# Advantages

The proposed framework provides several advantages:

- Lightweight model
- Fast inference
- Small storage requirement
- Automatic pothole localization
- Bounding box visualization
- Confidence-based predictions
- Easy model deployment
- Structured experimental outputs
- Automatic graph generation
- CSV-based prediction analysis
- Suitable for small datasets using transfer learning

---

# Limitations

The current implementation has several limitations.

The dataset contains only one road damage class:

```text
Pothole
```

Therefore, the current model should not be interpreted as a complete multi-class road damage detector.

It does not currently distinguish between:

```text
Longitudinal Cracks
Transverse Cracks
Alligator Cracks
Potholes
Surface Deterioration
```

Another limitation is the relatively small dataset size.

The dataset contains around 665 images, meaning performance may change when tested in completely different:

- Road environments
- Countries
- Weather conditions
- Camera devices
- Lighting conditions
- Driving speeds

---

# Future Work

Several improvements can be explored in future research.

## Multi-Class Road Damage Detection

Future datasets could contain:

```text
Potholes
Longitudinal Cracks
Transverse Cracks
Alligator Cracks
Road Surface Deterioration
```

## Model Comparison

YOLOv8n can be compared with:

```text
YOLOv5n
YOLOv8s
YOLOv10n
YOLOv11n
MobileNet-SSD
EfficientDet
Faster R-CNN
```

This would strengthen the experimental comparison.

## Model Optimization

Additional optimization methods could include:

```text
Quantization
Model Pruning
Knowledge Distillation
TensorRT Optimization
ONNX Conversion
```

These techniques could further reduce inference time and model size.

## Edge Deployment

The trained detector could be tested on:

```text
Raspberry Pi
NVIDIA Jetson
Smartphones
Vehicle Cameras
Drones
Embedded AI Systems
```

## Real-Time Video Detection

The system could be extended from images to:

```text
Road Video
Dashcam Video
CCTV Streams
Drone Footage
```

allowing real-time road condition monitoring.

## GPS-Based Road Damage Mapping

A future version could combine pothole detection with GPS coordinates.

```text
Camera Image
      +
Pothole Detection
      +
GPS Position
      =
Road Damage Map
```

This could be useful for smart city road maintenance systems.

---

# Applications

Potential applications include:

- Smart road maintenance
- Automated highway inspection
- Smart city infrastructure
- Municipal road monitoring
- Vehicle-based pothole detection
- Autonomous driving support
- Drone-based road inspection
- Road condition mapping
- Infrastructure maintenance planning

---

# Results Reporting Template

After training, the actual experimental results can be reported using:

| Metric | Experimental Result |
|---|---:|
| Precision | Add result |
| Recall | Add result |
| mAP@0.50 | Add result |
| mAP@0.50:0.95 | Add result |
| Model Size (MB) | Add result |
| Average Inference Time (ms) | Add result |
| Approximate FPS | Add result |
| Mean Absolute Count Error | Add result |

> Replace these placeholders only with values generated by the actual experiment.

---

# Reproducibility

The project uses:

```python
seed=42
deterministic=True
```

For reproducibility, the experimental environment should report:

- Python version
- PyTorch version
- Ultralytics version
- CPU
- GPU
- CUDA version
- Operating system
- Training time
- Random seed

---

# Project Information

```text
Project:
Lightweight Deep Learning for Road Surface Damage Detection

Deep Learning Model:
YOLOv8n

Task:
Object Detection

Detected Class:
Pothole

Dataset Size:
Approximately 665 images

Framework:
Ultralytics YOLO

Programming Language:
Python

Development Environment:
Jupyter Notebook
```

---

# Keywords

```text
Deep Learning
Computer Vision
Road Damage Detection
Pothole Detection
YOLOv8
YOLOv8n
Object Detection
Lightweight Deep Learning
Transfer Learning
Smart Transportation
Road Monitoring
Edge AI
Intelligent Transportation Systems
```

---

# Citation

If this project is used as part of a research publication, appropriate citations should be included for:

- YOLO / Ultralytics
- PyTorch
- The original pothole dataset
- Related road damage detection research

The exact dataset citation should be taken from the original dataset source.

---

# License

The licensing terms of the original road damage dataset should be followed when redistributing the dataset or publishing derived research.

The source code developed for this project may be released under an appropriate open-source license such as the **MIT License**, depending on the author's requirements.

---

# Conclusion

This project demonstrates a lightweight deep learning framework for detecting potholes from road images using **YOLOv8n**.

The system performs pothole localization, confidence estimation, validation, testing, prediction analysis, and computational performance evaluation.

The use of YOLOv8n provides a practical balance between **detection performance and computational efficiency**.

In addition to standard object detection metrics, the project analyzes model size, inference speed, FPS, pothole counts, prediction confidence, and bounding box characteristics.

The primary visualization:

```text
confidence_vs_bbox_area.png
```

provides additional insight into the relationship between the physical size of detected pothole regions and model confidence.

The resulting framework provides a foundation for future research involving **multi-class road damage detection, edge deployment, real-time video analysis, model optimization, GPS-based road damage mapping, and intelligent road maintenance systems**.
