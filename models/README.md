# Model files

This directory contains model metadata, training logs, epoch histories and validation metrics.
Model weights (`*.pth`) are ignored by Git.

## Deployed models

- `ethiopia_washed_custom_Noback_best_model`
- `honduras_natural_custom_Noback_best_model`

Kenya Natural remains work in progress and has no published model.

## Configuration

Each deployed weight has a matching `*_best_model.json`:

```json
{
  "bean_type": "ethiopia_washed",
  "architecture": "custom",
  "img_size": 128,
  "classes": ["bad", "good"],
  "model_file": "ethiopia_washed_custom_Noback_best_model.pth",
  "metrics_file": "ethiopia_washed_metrics.json",
  "display_name": "Ethiopia Washed - Custom CNN (Noback)"
}
```

Supported architectures are `resnet18`, `convnext_tiny`, `custom` and `ultrafast`.
The class order must match the output order used during training.

## Loading weights

At startup, `app.py` downloads each matching `.pth` file from
`dada8173/coffee-bean-classifier-models` using pinned commit
`c859bc6049e24649ca86bf4f5d2600d0fbec6198`. `HF_MODEL_REVISION` can override
the default when intentionally deploying another revision. Downloads use Hugging
Face's version-aware local cache. If the Hub is unavailable, the application
falls back to a matching local weight; set `USE_LOCAL_MODELS=1` to force that
local path during development.

After adding or changing a configuration or weight, restart the Flask/Gunicorn
process. Refreshing the browser alone does not rescan the directory.

## Results artifacts

- `*_training.log`: concise per-epoch console log.
- `*_history.json`: train/validation loss and accuracy for every epoch.
- `*_metrics.json`: aggregate and per-class metrics plus the confusion matrix.

The current metrics preserve the project's existing random 80/20 split of the
offline-augmented dataset. Read the limitations in the root README before using
the values as a benchmark.
