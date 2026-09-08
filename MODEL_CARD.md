---
library_name: pytorch
pipeline_tag: image-classification
tags:
  - coffee
  - computer-vision
  - binary-classification
---

# Coffee Bean Classifier Models

PyTorch checkpoints for the [Good-or-Bad-CoffeeBeans](https://github.com/dada8173/Good-or-Bad-CoffeeBeans) project.

## Available checkpoints

| Bean type | Checkpoint | Validation accuracy | Macro F1 |
|---|---|---:|---:|
| Ethiopia washed | `ethiopia_washed_custom_Noback_best_model.pth` | 79.02% | 78.72% |
| Honduras natural | `honduras_natural_custom_Noback_best_model.pth` | 78.07% | 70.76% |

Kenya is still under development and is not included.

## Model and preprocessing

- Architecture: project-specific CNN implemented in PyTorch
- Input: RGB image resized to 128 x 128
- Normalization: mean `(0.5, 0.5, 0.5)`, standard deviation `(0.5, 0.5, 0.5)`
- Output classes: `bad`, `good`

The matching `*_best_model.json` files contain the configuration used by the application. Evaluation summaries and training histories are included as JSON files.

## Evaluation caveat

The reported results come from the project's current 80/20 validation split. Offline-augmented images were split after augmentation, so variants derived from the same original crop may occur on both sides of the split. The validation set was also used for early stopping. These figures describe the recorded validation run and should not be treated as leakage-free, independent-test benchmarks.

## Dataset

The dataset consists of privately collected and manually labelled coffee-bean images. It is not included in this repository and is not publicly released.

## Intended use and limitations

These checkpoints are intended for demonstration and experimentation with the accompanying application. They cover only Ethiopia washed and Honduras natural beans from the project's private collection. Performance may not transfer to other origins, cameras, lighting conditions, roast levels, or production environments.

## License

No model-weight license has been declared yet. The source-code repository is licensed separately under MIT.
