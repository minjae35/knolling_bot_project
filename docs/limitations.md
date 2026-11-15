# Limitations

The published figure, 540/640 (84.4%), is target-pose prediction on synthetic RGB. No robot arm has executed these poses, and this repository does not contain a grasp or motion planner.

Spacing is the main residual error. The checker uses each object's rotated rectangle. A chamfer or capsule occupies fewer pixels than that rectangle, so the outline in the image does not determine the rectangle the metric uses.

Detection can still drop or duplicate an object. The system stops at eight objects. Instructions outside the three templates are not a supported task. `experiments/end_to_end/demo.py` only renders a full arrangement.

The 640-scene holdout and the official 2,000-scene test are closed. Further training needs a new development set and a separate unused final test.

Weights and both datasets are not in git. A clone of the repository cannot reload the model until those files are placed at the paths in [v2.0.0.md](v2.0.0.md). This note does not record a checksum-verified copy of the weights outside the machine that holds them. The instance disk is not a persistent volume.

Cross-machine image bytes are not guaranteed. Blender is external, and the dataset note does not promise bitwise PNG matches on another version.

Dataset acceptance and prediction scoring use different tolerances. The 100-scene pilot and the published 20,000-scene set do not use the same teacher. Both distinctions are in [dataset-contract.md](dataset-contract.md).
