# ML Accuracy Pipeline

This folder contains the machine-learning accuracy pipeline for the Pokémon Card AI Grader.

## Goal

Move from rule-based grading toward a data-trained PSA-style pre-grading model.

## Important Principle

The model will only become useful when trained on consistent, high-quality labeled examples.

Recommended labels:

- Actual PSA grade
- Card name
- Set name
- Cert number, if available
- Front scan path
- Back scan path
- Known defects
- Centering notes
- Corner notes
- Edge notes
- Surface notes

## Recommended Dataset Standard

- 1200 DPI PNG/TIFF scans for normal training
- 2400 DPI PNG/TIFF scans for high-quality reference samples
- Front and back scan required
- No scanner enhancement
- No auto-sharpening
- No dust removal
- No color correction
- Clean scanner glass

## Folder Structure

```text
ml/
  data/
    raw/
      front/
      back/
    labels/
      grading_labels.csv
    processed/
      features.csv
  scripts/
    extract_features.py
    train_baseline_model.py
  models/
```

## Training Stages

### Stage 1: Baseline Model

Uses extracted numeric features:

- centering score
- corner whitening ratios
- edge whitening ratios
- surface texture score
- scan quality values

Model type:

- Random Forest Regressor / Classifier

### Stage 2: Defect Segmentation

Train computer vision models to detect:

- whitening
- scratches
- dents
- print lines
- corner damage

### Stage 3: Grade Calibration

Use PSA-labeled outcomes to calibrate the final predicted grade.
