# daugx

A general-purpose multimodal data augmentation library for Python.

## Project Overview

daugx enables building seeded, reproducible augmentation pipelines that
operate across modalities (images, text, annotations, video, audio).
The core idea: define augmentation workflows as DAGs, visualize how they
affect dataset characteristics, and iteratively design pipelines that
target specific training data distributions.

### Key Concepts

- **Pipeline**: A DAG of augmentation nodes defined via Python API.
  Built with `Pipeline`, compiled into a callable `CompiledPipeline`.
- **Dataset**: A collection of `Sample`s conforming to a `Schema`.
  Zero I/O at construction — loading is deferred to materialization.
- **Schema**: Pure structural definition (dict) describing what
  components a sample has. User-declared. Predefined schemas for
  common datasets (COCO, VOC) planned for the future.
- **Sample**: One augmentation unit in preloaded state. Holds
  `Component`s by keyword. User builds samples in a loop.
  `sample.materialize()` loads heavy data and returns a `DataPackage`.
- **Component**: Standalone data holder. Types: `Image` (heavy,
  preloaded → materialized via cv2), `Label`, `BoundingBox`,
  `Polygon`, `KeyPoint` (all lightweight, born materialized).
- **DataPackage**: A fully materialized `Sample`. All heavy data
  loaded into memory. Transforms will receive `DataPackage` instances.
- **Node**: A handle returned by builder methods (`input()`, `then()`,
  `split()`, `merge()`). Records user intent, delegates to `Pipeline`.
- **Seeded reproducibility**: A global RNG (`np.random.default_rng(seed)`)
  is threaded through all operations.  Same seed = identical results.
- **Multi-input augmentations**: Transforms like Mosaic (4 inputs) and
  MixUp (2 inputs) declare an `inflation` ratio. `pipeline.merge()`
  validates input count at construction time.
- **Preloading**: Lightweight data (labels, coordinates, metadata) is
  always loaded. Heavy data (images, video, audio) stays as paths
  until `Sample.materialize()` is called.
- **Dataset characteristics** (planned): Geometric stats, class balance,
  and image properties computed and visualized to understand
  augmentation effects.

### Probability System

Two independent layers:

- **ext_exe_prob** (external execution probability): Determined by
  `split()` shares. Product of all branch shares from input to node.
  Controls which output path is selected.
- **int_exe_prob** (internal execution probability): The `p` parameter
  on `then()`. Per-node coin flip at runtime — transform fires with
  probability `p`, otherwise data bypasses it unchanged.
- **effective_prob** = ext_exe_prob * int_exe_prob.

Query after compile: `pipeline.get_ext_exe_prob(node_id)`,
`pipeline.get_int_exe_prob(node_id)`,
`pipeline.get_effective_prob(node_id)`.

### Coordinate Convention

Standard image coordinates: origin at top-left, x-axis rightward,
y-axis downward (OpenCV/PIL/numpy convention).

## Code Style

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):

- Google-style docstrings for all public modules, classes, functions
- Type annotations on all function signatures
- `snake_case` for functions/variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for constants
- Max line length: 80 characters
- Imports ordered: stdlib, third-party, local (blank line separated)

## Development Workflow

**Test-Driven Development (TDD)**: Write tests first, then implement.

1. Write a failing test that defines the expected behavior
2. Implement the minimum code to make the test pass
3. Refactor while keeping tests green

No production code without a corresponding test.

## Development

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Python Version

Python 3.10+

### Build System

setuptools with pyproject.toml

### Testing

```bash
pytest                    # run all tests
pytest tests/ -v          # verbose
pytest tests/test_X.py    # specific test file
```

Framework: **pytest**

### Dependencies

Core: numpy, opencv-python, scipy, pyyaml, xmltodict

### Project Structure

```
daugx/
  __init__.py              # Re-exports: Pipeline, Dataset, Schema, Sample,
                           #   DataPackage, Image, Label, BoundingBox,
                           #   Polygon, KeyPoint, Node, CompiledPipeline
  core/
    pipeline.py            # Pipeline builder (DAG construction, compile, visualization)
    dataset.py             # Dataset (schema + samples collection, validation)
    node.py                # Node handle (then, split — delegates to Pipeline)
    compiled_pipeline.py   # CompiledPipeline (trace, path selection, execution)
    augmentation/          # Transforms, annotations, boundaries, borders
    data/
      component.py         # Component ABC, ComponentState enum
      schema.py            # Schema (structural definition, validation)
      sample.py            # Sample (preloaded holder, materialize → DataPackage)
      data_package.py      # DataPackage (materialized sample)
      components/
        __init__.py         # Re-exports all component types
        image.py            # Image (heavy: path → np.ndarray via cv2)
        label.py            # Label (lightweight: class_id + name)
        bounding_box.py     # BoundingBox (lightweight: (2,2) min/max)
        polygon.py          # Polygon (lightweight: (n,2) points)
        keypoint.py         # KeyPoint (lightweight: x, y, visibility)
  utils/                   # Misc utilities, visualization
  tests/
    test_pipeline.py       # Pipeline API, probability, visualization, probabilistic tests
    test_data_loading.py   # Components, Schema, Sample, DataPackage, Dataset tests
    test_annotations.py    # Annotation/boundary tests
    test_augmentations.py  # Transform tests
    test_borders.py        # ImageBorder tests
    test_boundaries.py     # Boundary type tests
  errors/                  # Custom exceptions (SchemaValidationError)
  _legacy/                 # Old code preserved for reference; delete when no longer needed
```

## Architecture Notes

### Pipeline API

```python
import daugx

pipeline = daugx.Pipeline(seed=42)

# Define schema — pure structure, no data
schema = daugx.Schema({
    "image": daugx.Image,
    "objects": [daugx.BoundingBox, daugx.Label],
})

# Build samples in a loop (user handles loading)
samples = []
for img_path, bboxes, labels in my_data:
    s = daugx.Sample(
        image=daugx.Image(path=img_path),
        objects=[
            (daugx.BoundingBox(bb), daugx.Label(lbl))
            for bb, lbl in zip(bboxes, labels)
        ],
    )
    samples.append(s)

# Dataset validates all samples against schema
ds = daugx.Dataset(schema=schema, samples=samples,
                   name="COCO")

# Build DAG
inp = pipeline.input(ds)
a, b = inp.split(0.7, 0.3)
a.then(Resize(640, 640)).then(Shift(10, 0))
b.then(Rotate(15), p=0.8)

# Compile and use
fetch = pipeline.compile()
print(pipeline)  # ASCII tree visualization
result = fetch()  # Returns DataPackage
```

### Pipeline Visualization

`print(pipeline)` renders the DAG as an ASCII tree:

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

Merge nodes render as separate sections with `← [sources]` notation.

### RNG Discipline

All randomness MUST use `np.random.default_rng()`. Never use `random`
module or `np.random.seed()`. The RNG instance is passed explicitly to
all functions that need randomness.

### Transform Types

- **SITransform**: Single-input augmentations (inflation=1)
- **MITransform**: Multi-input augmentations (inflation < 1.0)
- **IOTransform**: Image-only transforms (no annotation handling)

All transforms implement `_apply_on_image()` and `_apply_on_annots()`
to keep image and annotation transformations synchronized.

### Configuration Style

Python-first API. Pipelines are defined programmatically in code.
JSON/YAML serialization may be added later for workflow sharing.
