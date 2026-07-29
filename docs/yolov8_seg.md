# YOLOv8n-seg deployment and evaluation

`yolov8n-seg.pt` is the smallest and least accurate official YOLOv8
instance-segmentation checkpoint. The adapter in `src/detectors/yolov8_seg.py`
returns boxes, class IDs, confidence scores, and full-resolution boolean masks.
It can also convert these outputs to Detectron2 `Instances`, which keeps it
compatible with the current policy-side consumers.

The evaluation entry point enables `SEMANTIC_CURIOSITY_LIGHT_IMPORT=1` so an
offline detector run does not initialize Habitat and all policy agents.

The evaluator reports the same standard COCO AP/AR families commonly produced
by Detectron2's `COCOEvaluator`, independently for bounding boxes and instance
masks. Dataset categories are matched to model categories by name. This is
important because the repository dataset uses contiguous IDs 0-4 while the
pretrained COCO model uses IDs 56, 57, 59, 61, and 72.

`metrics.json` contains both overall metrics and a `per_category` section under
`bbox` and `segm`. The per-category section reports the same AP/AR family for
chair, couch, bed, toilet, and refrigerator.

Run a small smoke evaluation:

```bash
conda run -n explore python tools/evaluate_yolov8_seg.py --limit 8
```

Run the complete embodied test set:

```bash
conda run -n explore python tools/evaluate_yolov8_seg.py
```

The first run downloads `yolov8n-seg.pt` if it is not already cached. Results
are written under `outputs/yolov8n_seg_embodied_test/`:

- `metrics.json`
- `bbox_predictions.json`
- `segmentation_predictions.json`

Configuration is in `configs/yolov8_seg/eval_embodied.yaml`. A local or
fine-tuned checkpoint can be selected with `--weights /path/to/best.pt`.

## Training from Detectron2 COCO datasets

The training configuration is `configs/yolov8_seg/train_embodied.yaml`. It
converts Detectron2-compatible COCO RLE masks into YOLO polygons, creates image
symlinks instead of image copies, writes `data.yaml`, and starts Ultralytics
training.

Check conversion on a small subset without training:

```bash
conda run -n explore python tools/train_yolov8_seg.py \
  --prepare-only --limit 8
```

`--limit` data is only for smoke tests. Running again without `--limit`
prepares the complete splits and automatically invalidates Ultralytics' label
cache.

Prepare the full dataset and train:

```bash
conda run -n explore python tools/train_yolov8_seg.py
```

Reuse prepared data:

```bash
conda run -n explore python tools/train_yolov8_seg.py \
  --skip-prepare --epochs 100 --batch 8 --device 0
```

The default train split is Detectron2's `embodied_frontier_epi20`; validation
uses `embodied_test`. Any other COCO instance dataset can be selected by
changing the annotation and image-root paths. The train and validation category
names and order must match. W&B logging is disabled by default because the
repository's Ultralytics 8.0.135 callback is incompatible with filesystem paths
in the `project` option; local CSV, plots, TensorBoard events, and checkpoints
are still produced.
