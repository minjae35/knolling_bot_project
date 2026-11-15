# Third-party notices

This repository has no `LICENSE` file. Nothing here grants a license to reuse the project code. Procedural assets store `license: CC0-1.0` on the asset record. That is the default of the `license` field in `src/knolling_dataset/schema.py`, not a license for the Python code.

The packages below were read from the environment that runs v2 (`/venv/knolling`). Versions match `requirements-neural-lock.txt`.

| Package | Version in this environment | License metadata |
| --- | --- | --- |
| torch | 2.9.1+cu128 | BSD-3-Clause |
| torchvision | 0.24.1+cu128 | BSD |
| transformers | 4.57.3 | Apache-2.0 |
| huggingface_hub | 0.36.2 | Apache |
| numpy | 2.3.5 | BSD (OSI classifier) |
| scipy | 1.16.3 | BSD (OSI classifier) |
| shapely | 2.1.2 | BSD-3-Clause |
| pillow | 12.0.0 | MIT-CMU |
| pytest | 9.0.1 | MIT |
| sentencepiece | 0.2.1 | not stated in the installed metadata |

The local RT-DETR snapshot `PekingU/rtdetr_r18vd_coco_o365` at `386c438` includes a model card whose header says `license: apache-2.0`. The MiniLM snapshot was downloaded without its model card, so this tree does not verify that weight's license. Blender, used only as an external renderer, is GPL-2.0-or-later and is not included here.

RT-DETR is Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024. The text encoder follows Reimers and Gurevych, Sentence-BERT, EMNLP 2019.
