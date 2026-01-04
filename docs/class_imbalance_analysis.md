# Class Imbalance Analysis and Solution

## The Problem

Our bus occupancy dataset has severe class imbalance:

| Class | Description | % of Data |
|-------|-------------|-----------|
| 0 | EMPTY | 72.3% |
| 1 | MANY_SEATS_AVAILABLE | 26.3% |
| 2 | FEW_SEATS_AVAILABLE | 1.0% |
| 3 | STANDING_ROOM_ONLY | 0.4% |

Classes 2 and 3 (the crowded buses we want to predict) make up only **1.4%** of the data.

### Initial Model Results (No Class Weighting)

```
Accuracy: 75.9%
Confusion Matrix:
[[695374 164465      0      0]
 [109338 238526      0      0]
 [  3061  13827      0      0]
 [   452   5278      0      0]]
```

The model achieved 76% accuracy by **never predicting classes 2 or 3**. It learned to always predict "not crowded" because that's correct 98.6% of the time. This makes the model useless for its intended purpose.

## The Solution

### 1. Balanced Class Weights (sklearn)

We first apply sklearn's `compute_class_weight('balanced')` which calculates weights inversely proportional to class frequency:

```python
# Calculated weights:
{0: 0.35, 1: 0.95, 2: 25.8, 3: 59.4}
```

This gives class 3 samples ~170x more importance than class 0 samples during training.

### 2. Additional Multipliers for Severe Imbalance

The balanced weights alone weren't aggressive enough. We added custom multipliers:

```python
CLASS_WEIGHT_MULTIPLIER = {
    0: 1.0,   # EMPTY - baseline
    1: 2.0,   # MANY_SEATS - slight boost
    2: 10.0,  # FEW_SEATS - significant boost
    3: 20.0,  # STANDING - heavy boost
}
```

Final adjusted weights:
```python
{0: 0.35, 1: 1.9, 2: 258.3, 3: 1188.1}
```

This makes a single class 3 sample worth ~3,400x a class 0 sample.

### 3. Model Configuration Changes

```python
XGBOOST_PARAMS = {
    "objective": "multi:softprob",  # Probabilities instead of hard predictions
    "max_depth": 8,                  # Deeper trees (was 7)
    "learning_rate": 0.05,           # Faster learning (was 0.02)
    "n_estimators": 200,             # More trees (was 150)
    "gamma": 0.1,                    # Regularization to prevent overfitting
}
```

## Results After Fix

```
Accuracy: 51.7%
Confusion Matrix:
[[504667 177221  70406 107545]
 [  3972 121577  80509 141806]
 [    54   2262   6283   8289]
 [     0    913   1228   3589]]
```

### Per-Class Recall Comparison

| Class | Before | After | Change |
|-------|--------|-------|--------|
| 0 (EMPTY) | 81% | 59% | -22% |
| 1 (MANY_SEATS) | 69% | 35% | -34% |
| 2 (FEW_SEATS) | **0%** | **37%** | +37% |
| 3 (STANDING) | **0%** | **63%** | +63% |

## Why This Tradeoff is Worth It

1. **The model now serves its purpose**: Users want to know when buses are crowded. A model that never predicts crowding is useless.

2. **False positives are acceptable**: If we predict "crowded" when it's actually "many seats", the user might wait for the next bus or stand when they could sit. Minor inconvenience.

3. **False negatives are worse**: If we predict "empty" when it's actually "standing room only", users make decisions based on bad information and end up in uncomfortable situations.

4. **Class 3 has 63% recall**: We correctly identify nearly 2/3 of the most crowded situations.

## Alternative Approaches (Not Implemented)

1. **SMOTE/Oversampling**: Generate synthetic samples for rare classes. Can help but may create unrealistic data points.

2. **Undersampling**: Remove majority class samples. Loses potentially useful information.

3. **Threshold Tuning**: Since we use `multi:softprob`, we could lower the probability threshold for predicting classes 2-3. This would trade precision for recall.

4. **Cost-Sensitive Learning**: Define custom loss functions that penalize misclassifying crowded buses more heavily.

5. **Ensemble Methods**: Train separate binary classifiers (crowded vs not-crowded) and combine.

## Conclusion

By applying aggressive class weighting, we transformed a model that was accurate but useless into one that actually predicts what users care about: crowded buses. The 52% overall accuracy looks worse on paper, but the model is now fit for purpose.

The key insight: **accuracy is the wrong metric for imbalanced classification**. Per-class recall on the minority classes is what matters for this use case.
