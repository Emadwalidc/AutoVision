"""Pluggable detector, multi-task classifier, retriever, and orchestration pipeline."""

from backend.ml.classifier import AttributeClassifier, ClassifierBatchOutput, PyTorchAutoVisionClassifier
from backend.ml.detector import DetectedBox, VehicleDetector, YoloVehicleDetector
from backend.ml.pipeline import AutoVisionPipeline
from backend.ml.retriever import NumpyCosineRetriever, VehicleRetriever

__all__ = [
    "AttributeClassifier",
    "AutoVisionPipeline",
    "ClassifierBatchOutput",
    "DetectedBox",
    "NumpyCosineRetriever",
    "PyTorchAutoVisionClassifier",
    "VehicleDetector",
    "VehicleRetriever",
    "YoloVehicleDetector",
]
