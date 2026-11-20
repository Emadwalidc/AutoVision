/**
 * TypeScript contract mirroring AutoVision FastAPI `backend/schemas.py` 1-to-1,
 * plus lean consumer app navigation and history state types.
 */

export interface HealthResponse {
  status: string;
  model_version: string;
  device: string;
  ready: boolean;
  num_makes: number;
  num_models: number;
  num_prototype_classes: number;
  num_bodies: number;
  num_colors: number;
  gallery_size: number;
  yolo_available: boolean;
}

export interface CatalogColor {
  label: string;
  group: string;
  hex: string;
}

export interface CatalogResponse {
  model_version: string;
  makes: string[];
  models: string[];
  prototype_classes: string[];
  body_types: string[];
  colors: CatalogColor[];
  year_min: number;
  year_max: number;
}

export interface ImageSize {
  width: number;
  height: number;
}

export interface BoundingBoxXYXY {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface BoundingBoxNorm {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TopKPrediction {
  label: string;
  display_name: string;
  confidence: number;
}

export interface FewShotMatch {
  canonical_class: string;
  display_name: string;
  make: string;
  body_type: string;
  similarity: number;
  is_few_shot: boolean;
}

export interface MakePrediction {
  label: string;
  display_name: string;
  confidence: number;
  top3: TopKPrediction[];
}

export interface ModelPrediction {
  canonical_class: string;
  display_name: string;
  confidence: number;
  top3: TopKPrediction[];
  few_shot_match: FewShotMatch | null;
}

export interface YearPrediction {
  predicted: number;
  predicted_rounded: number;
  range_low: number;
  range_high: number;
  std: number;
}

export interface BodyTypePrediction {
  label: string;
  display_name: string;
  confidence: number;
  top3: TopKPrediction[];
}

export interface ColorPrediction {
  label: string;
  display_name: string;
  group: string;
  hex: string;
  confidence: number;
  top3: TopKPrediction[];
}

export interface VehicleAttributes {
  make: MakePrediction;
  model: ModelPrediction;
  year: YearPrediction;
  body_type: BodyTypePrediction;
  color: ColorPrediction;
}

export interface SimilarVehicle {
  rel_path: string;
  canonical_class: string;
  display_name: string;
  make: string;
  year: number | null;
  body_type: string;
  similarity: number;
  thumbnail_url: string;
}

export interface VehicleDetectionResult {
  vehicle_index: number;
  detected_by_yolo: boolean;
  detection_confidence: number;
  area_ratio: number;
  bbox_xyxy: BoundingBoxXYXY;
  bbox_norm: BoundingBoxNorm;
  attributes: VehicleAttributes;
  similar_vehicles: SimilarVehicle[];
}

export interface AnalyzeTimingsMs {
  decode_ms: number;
  detect_ms: number;
  classify_ms: number;
  retrieve_ms: number;
  total_ms: number;
}

export interface AnalyzeResponse {
  model_version: string;
  device: string;
  image_size: ImageSize;
  num_vehicles: number;
  primary_vehicle_index: number;
  vehicles: VehicleDetectionResult[];
  timings_ms: AnalyzeTimingsMs;
}

export interface ErrorResponse {
  error_code: string;
  message: string;
  detail?: string | null;
}

export type TabId = "scan" | "garage";

export interface ScanHistoryItem {
  id: string;
  createdAt: string;
  imageUri: string;
  response: AnalyzeResponse;
}
