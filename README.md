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
import numpy as np
from daugx.augmentations import Resize, Shift, Rotate

# 1. Build samples — pass components directly
samples = [
    daugx.Sample(
        daugx.Image(path="/data/coco/img001.jpg"),
        daugx.ImageBoundingBox(
            np.array([[10, 20], [100, 200]]),
            class_id=0,
            class_name="cat",
        ),
    ),
    # ... more samples ...
]

# 2. Create a dataset
coco = daugx.Dataset(samples=samples, name="COCO")

# 3. Build an augmentation pipeline as a DAG
pipeline = daugx.Pipeline(seed=42)
inp = pipeline.input(coco)

# Branch: 70% resize+shift, 30% rotate
a, b = inp.split(0.7, 0.3)
a.then(Resize(640, 640)).then(Shift(x_shift=10))
b.then(Rotate(angle=15), p=0.8)   # p=0.8 → fires 80%, bypassed 20%

# 4. Compile and visualize
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
coco = daugx.Dataset(samples=coco_samples, name="COCO")
voc  = daugx.Dataset(samples=voc_samples,  name="VOC")

pipeline = daugx.Pipeline(seed=42)
c1 = pipeline.input(coco).then(Resize(640, 640))
c2 = pipeline.input(coco).then(Resize(640, 640))
v1 = pipeline.input(voc).then(Resize(640, 640)).then(Rotate(15))
v2 = pipeline.input(voc).then(Resize(640, 640)).then(Flip())

# Combine 4 independently augmented streams into a mosaic
out = pipeline.merge([c1, c2, v1, v2], Mosaic())
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
out = pipeline.merge(inputs, Mosaic())  # 4 → 1
```

### Pipeline Visualization

`print(pipeline)` renders the full DAG as an ASCII tree, showing branch
shares, execution probabilities, and ext_exe_prob on output nodes.

### Data Loading

Data loading uses a Sample/DataPackage architecture:

1. **Sample** holds preloaded components (paths + lightweight data)
2. **DataPackage** is a materialized sample with all data loaded

Components are passed directly to `Sample` as positional arguments.
All annotation types are first-class `Component` subclasses — there is
no wrapper class. Retrieve components by type with `get()` and
`get_all()`, or get all annotations with `get_annotations()`.

```python
# Build a sample with flat annotation components
sample = daugx.Sample(
    daugx.Image(path="/data/img.jpg"),
    daugx.ImageBoundingBox(
        bbox, class_id=0, class_name="cat",
    ),
    daugx.Constant(value=1234, name="image_id"),
)

# Materialize: loads heavy data (images) on demand
package = sample.materialize()
package.get(daugx.Image).data            # np.ndarray (H, W, C)
package.get_all(daugx.ImageBoundingBox)  # all bbox annotations
package.get_annotations()                # all annotation components
package.get(daugx.Constant, name="image_id").value  # 1234
```

Components are either **heavy** (`Image` — starts preloaded, loads via
cv2 on `materialize()`) or **lightweight** (all annotation types,
`Text`, `Constant` — always materialized, no I/O).

### Annotation Types

Annotations are `Component` subclasses that carry a `target` field
linking them to their parent component's `name`, plus `class_id` and
`class_name` where applicable.

| Type | Description |
|------|-------------|
| `ImageBoundingBox(points, class_id, class_name, target, name)` | Axis-aligned bounding box, (2,2) min/max |
| `ImagePolygon(points, class_id, class_name, target, name)` | Polygon boundary, (n,2) points |
| `ImageKeyPoint(x, y, visibility, class_id, class_name, target, name)` | Single keypoint |
| `ImageCategory(class_id, class_name, target, name)` | Image-level category label |

All spatial annotation types (`ImageBoundingBox`, `ImagePolygon`,
`ImageKeyPoint`) support `shift()`, `scale()`, `rotate()`, `clip()`,
and `is_valid()`. These methods return new instances and propagate
`target`, `class_id`, and `class_name`.

### Multimodal Samples

For datasets with multiple modalities, use `name` to identify which
modality a component belongs to, and `target` to link annotations to
their parent component.

```python
sample = daugx.Sample(
    daugx.Image(path="frame.jpg", name="video"),
    daugx.ImageBoundingBox(
        bb, class_id=0, class_name="cat", target="video",
    ),
    daugx.ImageCategory(1, "scene", target="video"),
)

# Retrieve by modality
sample.get(daugx.Image, name="video")
sample.get_annotations(target="video")

# Discover which modalities are present
sample.modalities()  # {"video"}
```

### Constants

Use `Constant` to attach metadata that must survive augmentation
unchanged — image ids, split names, confidence scores, etc.

```python
sample = daugx.Sample(
    daugx.Image(path="img.jpg"),
    daugx.Constant(value=42, name="image_id"),
    daugx.Constant(value="train", name="split"),
)
```

### Merging Samples

Combine two samples into one with `merge()` (or `+`). Returns a new
Sample; neither original is mutated.

```python
merged = sample_a.merge(sample_b)
merged = sample_a + sample_b  # equivalent
```

The same methods are available on `DataPackage`.

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Pipeline(seed=None)` | DAG builder. Methods: `input()`, `merge()`, `compile()` |
| `Dataset(samples, name=None)` | Collection of samples |
| `Sample(*components)` | Preloaded augmentation unit. `materialize()`, `merge()`, `modalities()`, `get_annotations()` |
| `DataPackage` | Materialized sample. `get(type)`, `get_all(type)`, `get_annotations()`, `replacing(old, new)`, `merge()`, `modalities()` |
| `Annotation` | Abstract base for all annotation components. Adds `target` field. |
| `Image(path, format_hint=None, name=None)` | Heavy component: path → pixels via cv2 |
| `Text(text, language="en", metadata=None, name=None)` | Lightweight: text content |
| `Constant(value, name=None)` | Lightweight: any immutable value |
| `ImageBoundingBox(points, class_id=None, class_name=None, target=None, name=None)` | Axis-aligned bounding box |
| `ImagePolygon(points, class_id=None, class_name=None, target=None, name=None)` | Polygon boundary |
| `ImageKeyPoint(x, y, visibility=None, class_id=None, class_name=None, target=None, name=None)` | Single keypoint |
| `ImageCategory(class_id, class_name=None, target=None, name=None)` | Image-level category label |
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
