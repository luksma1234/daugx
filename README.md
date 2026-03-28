# daugx

A general-purpose multimodal data augmentation library for Python.

Build seeded, reproducible augmentation pipelines as directed acyclic
graphs (DAGs). Combine multiple datasets, branch with probabilities,
merge with multi-input augmentations, and visualize the entire workflow.

## Getting Started

```bash
pip install daugx
```

## Quick Example

```python
import daugx
from daugx.augmentations import Resize, Shift, Rotate

# 1. Define a schema — what does one sample look like?
schema = daugx.Schema({
    "image": daugx.Image,
    "objects": [daugx.BoundingBox, daugx.Label],
})

# 2. Build samples (you control loading)
import numpy as np
samples = [
    daugx.Sample(
        image=daugx.Image(path="/data/coco/img001.jpg"),
        objects=[
            (daugx.BoundingBox(np.array([[10, 20], [100, 200]])),
             daugx.Label(0, "cat")),
        ],
    ),
    # ... more samples ...
]

# 3. Create a dataset — validates samples against schema
coco = daugx.Dataset(schema=schema, samples=samples, name="COCO")

# 4. Build an augmentation pipeline as a DAG
pipeline = daugx.Pipeline(seed=42)
inp = pipeline.input(coco)

# Branch: 70% resize+shift, 30% rotate
a, b = inp.split(0.7, 0.3)
a.then(Resize(640, 640)).then(Shift(x_shift=10))
b.then(Rotate(angle=15), p=0.8)   # p=0.8 → fires 80%, bypassed 20%

# 5. Compile and visualize
fetch = pipeline.compile()
print(pipeline)
```

```
COCO
└── split(0.7, 0.3)
    ├── Branch 1 (share=0.7)
    │   └── Resize
    │       └── Shift
    │           └── [Output]  ext=0.7
    └── Branch 2 (share=0.3)
        └── Rotate (p=0.8)
            └── [Output]  ext=0.3
```

```python
# Fetch augmented samples — data is materialized here, not before
result = fetch()  # Returns a DataPackage with loaded components
```

## Features

### Reproducibility

Every pipeline is initialized with a seed. Same seed = identical
augmentation sequence. If no seed is given, one is generated
automatically. Share your seed and pipeline definition to let anyone
reproduce your exact results.

```python
pipeline = daugx.Pipeline(seed=42)
# ... build pipeline ...
fetch = pipeline.compile()
fetch.reset()  # Reset RNG to reproduce from the start
```

### Multiple Datasets

Use data from different sources in the same pipeline. Each
`pipeline.input()` creates an independent data stream that can be
augmented separately before merging.

```python
schema = daugx.Schema({"image": daugx.Image})
coco = daugx.Dataset(schema=schema, samples=coco_samples, name="COCO")
voc  = daugx.Dataset(schema=schema, samples=voc_samples,  name="VOC")

pipeline = daugx.Pipeline(seed=42)
c1 = pipeline.input(coco).then(Resize(640, 640))
c2 = pipeline.input(coco).then(Resize(640, 640))
v1 = pipeline.input(voc).then(Resize(640, 640)).then(Rotate(15))
v2 = pipeline.input(voc).then(Resize(640, 640)).then(Flip())

# Combine 4 independently augmented streams into a mosaic
out = pipeline.merge([c1, c2, v1, v2], Mosaic(mode="resize"))
out.then(Normalize())
```

### Probability Control

Two layers of probability give fine-grained control:

- **Branch shares** (`split`): Control which path through the DAG is
  selected. `split(0.7, 0.3)` routes 70% of samples down branch 1.
- **Execution probability** (`p`): Per-node coin flip. `then(Aug(), p=0.4)`
  means the augmentation fires 40% of the time; 60% it's bypassed.

```python
# Query probabilities after compile
pipeline.get_ext_exe_prob(node.node_id)   # Branch selection probability
pipeline.get_int_exe_prob(node.node_id)   # Per-node fire probability
pipeline.get_effective_prob(node.node_id) # ext * int (overall probability)
```

### Multi-Input Augmentations

Augmentations like Mosaic (4 images → 1) and MixUp (2 images → 1) are
supported via `pipeline.merge()`. Input count is validated at
construction time.

```python
inputs = [pipeline.input(ds) for _ in range(4)]
out = pipeline.merge(inputs, Mosaic(mode="resize"))  # 4 → 1
```

### Pipeline Visualization

`print(pipeline)` renders the full DAG as an ASCII tree, showing branch
shares, execution probabilities, and ext_exe_prob on output nodes.

### Data Loading

Data loading uses a Schema/Sample/DataPackage architecture:

1. **Schema** defines structure — what components a sample has
2. **Sample** holds preloaded components (paths + lightweight data)
3. **DataPackage** is a materialized sample with all data loaded

```python
# Schema: pure structure, no data
schema = daugx.Schema({
    "image": daugx.Image,
    "objects": [daugx.BoundingBox, daugx.Label],
})

# Samples: user builds in a loop
sample = daugx.Sample(
    image=daugx.Image(path="/data/img.jpg"),
    objects=[(daugx.BoundingBox(bbox), daugx.Label(0, "cat"))],
)

# Materialize: loads heavy data (images) on demand
package = sample.materialize()
package["image"].data  # np.ndarray (H, W, C)
```

Components are either **heavy** (`Image` — starts preloaded, loads via
cv2 on `materialize()`) or **lightweight** (`Label`, `BoundingBox`,
`Polygon`, `KeyPoint` — always materialized, no I/O).

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Pipeline(seed=None)` | DAG builder. Methods: `input()`, `merge()`, `compile()` |
| `Schema(definition)` | Structural definition of a sample's components |
| `Dataset(schema, samples, name=None)` | Collection of samples validated against a schema |
| `Sample(**components)` | Preloaded augmentation unit. `materialize()` → `DataPackage` |
| `DataPackage` | Materialized sample with all data loaded |
| `Image(path, format_hint=None)` | Heavy component: path → pixels via cv2 |
| `Label(class_id, name=None)` | Lightweight: classification label |
| `BoundingBox(points)` | Lightweight: (2,2) min/max bounding box |
| `Polygon(points)` | Lightweight: (n,2) polygon boundary |
| `KeyPoint(x, y, visibility=None)` | Lightweight: single keypoint |
| `Node` | Handle returned by builder methods: `then()`, `split()` |
| `CompiledPipeline` | Callable returned by `compile()`: `__call__()`, `stream(n)`, `trace()`, `reset()` |

### Pipeline Methods

| Method | Description |
|--------|-------------|
| `pipeline.input(dataset)` | Register a dataset, returns `Node` |
| `pipeline.merge(nodes, transform)` | Combine N streams with a multi-input transform |
| `pipeline.compile()` | Freeze DAG, compute probabilities, return `CompiledPipeline` |

### Node Methods

| Method | Description |
|--------|-------------|
| `node.then(transform, p=1.0)` | Append augmentation, returns new `Node` |
| `node.split(*shares)` | Branch into N paths by probability, returns tuple of `Node`s |

## Python Version

Python 3.10+

## License

TBD
