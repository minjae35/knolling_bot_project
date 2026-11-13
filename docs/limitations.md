# Limitations

The published figure, 540/640 (84.4%), is target-pose prediction on synthetic
RGB. No robot arm has executed these poses, and this repository does not
contain a grasp or motion planner.

Spacing is the main residual error. The checker uses each object's rotated
rectangle. A chamfer or capsule occupies fewer pixels than that rectangle, so
the outline in the image does not determine the rectangle the metric uses.

Detection can still drop or duplicate an object. The system stops at eight
objects. Instructions outside the three templates are not a supported task.
`experiments/end_to_end/demo.py` only renders a full arrangement.

The 640-scene holdout and the official 2,000-scene test are closed. Further
training belongs on `dev/knolling-v3` and needs a new development set plus a
separate unused final test.

Weights and both datasets are not in git. Until a verified copy exists
outside this machine, another researcher cannot reload the model from the
repository alone. The external copy that was started has not been checked
against the SHA-256 list, so it is not recorded here as a finished backup.

Cross-machine image bytes are not guaranteed. Blender is external, and the
dataset note does not promise bitwise PNG matches on another version.
