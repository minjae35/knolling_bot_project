# Limitations

The published figure, 540/640 (84.4%), is target-pose prediction on synthetic RGB. No robot arm has executed these poses, and this repository does not contain a grasp or motion planner.

Spacing is the main residual error. The checker uses each object's rotated rectangle. A chamfer or capsule occupies fewer pixels than that rectangle, so the outline in the image does not determine the rectangle the metric uses.

Detection can still drop or duplicate an object. The system stops at eight objects. Instructions outside the three templates are not a supported task. `experiments/end_to_end/demo.py` only renders a full arrangement.

The 640-scene holdout and the official 2,000-scene test are closed. Further training needs a new development set and a separate unused final test.

The checkpoints and both datasets are not part of the git repository, and the repository does not distribute them. Reloading the published system requires every file listed in [v2.0.0.md](v2.0.0.md), including the blob directories named by those checkpoints. Without those files the source tree alone cannot reproduce the 540/640 measurement.

Rendered PNG bytes depend on an external Blender installation. Another machine or another Blender version can change the image bytes while leaving the scene records unchanged.

The 20,000-scene figure is the size of the generated corpus, not the number of scenes used in any one training stage. Dataset acceptance and prediction scoring also use different tolerances. Both distinctions are in [dataset-contract.md](dataset-contract.md).
