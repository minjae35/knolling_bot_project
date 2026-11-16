# Primary-source literature audit

Checked on 2026-11-15. `references.bib` contains eleven verified entries; citation use should be selective rather than padding the bibliography. All claims below are paraphrases, not quotations. Do not claim direct experimental comparison with these systems: the project's synthetic data and success predicate differ.

## Closest prior work and naming

**Hu et al., 2025, Knolling Bot: Teaching Robots the Human Notion of Tidiness**, NeurIPS 2025 Creative AI Track. Primary publication: https://papers.neurips.cc/paper_files/paper/2025/hash/91ae44cea7a1a23e24941b83d55427d7-Abstract-Creative_AI_Track.html ; full text: https://arxiv.org/html/2310.04566v3 . The original 2023 arXiv title was *Learning Robotic Object Arrangement from Tidy Demonstrations*; do not mix that old title and author list with the 2025 version. The latest paper represents objects by width/length, predicts placements sequentially with a transformer and mixture-density output, encodes arrangement preferences through ordering, and demonstrates a perception/arrangement/control pipeline on a physical robot. This is essential context for knolling and for separating this course project's results from external work. Key `hu2025knolling`.

**Hu, Zhang, and Lipson, 2023, Knolling bot 2.0: Enhancing Object Organization with Self-supervised Graspability Estimation**, NeurIPS Robot Learning Workshop. Primary: https://arxiv.org/abs/2310.19226 ; workshop PDF https://www.robot-learning.ml/2023/files/paper56.pdf . Introduces graspability estimation and separation of ungraspable clutter before rearrangement. This project's release label “v2” should explicitly mean its own implementation version, not this published system. The course project has no grasping/execution evaluation. Do not import the external paper's 95.7% graspability metric. Key `hu2023knolling2`.

Suggested compact paper wording: “Knolling Bot models tidiness through sequential object placement (Hu et al., 2025), while its graspability extension addresses physical clutter (Hu et al., 2023). Here, v2 denotes our course implementation: language-conditioned target-pose prediction on synthetic RGB, evaluated with explicit geometric and preservation constraints.” This distinguishes implementation/evaluation without claiming a code lineage that has not been established.

**Liu et al., 2022, StructFormer**, ICRA. Primary author pages: https://sites.google.com/view/structformer and https://research.nvidia.com/labs/srl/publication/liu-2022-struct-former/ ; paper https://arxiv.org/abs/2110.10189 ; author code https://github.com/wliu88/StructFormer . Predicts which objects to manipulate and where to place them from partial-view point clouds and structured language; physical robot evaluation. This is closer to language-conditioned arrangement than a generic VLA citation. Current project uses a single RGB view, limited instruction templates, 2-D poses and explicit row constraints. Key `liu2022structformer`.

## Vision and language components

**Carion et al., 2020, DETR**, ECCV. Primary proceedings https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/832_ECCV_2020_paper.php ; paper https://arxiv.org/abs/2005.12872 . Frames object detection as direct set prediction with learned object queries and bipartite matching. This FAIR paper is also an appropriate visual reference: restrained figure colors, concise diagrams, alignment, readable captions. Do not imply FAIR affiliation. Key `carion2020detr`.

**Zhao et al., 2024, RT-DETR**, CVPR. Primary proceedings and BibTeX https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html . Efficient hybrid encoder and query selection for real-time transformer detection. Cite for the adopted perception backbone. RT-DETR's original real-time benchmark is not the current project's throughput; no speed claim should transfer. Key `zhao2024rtdetr`.

**Wang et al., 2020, MiniLM**, NeurIPS 33. Primary https://proceedings.neurips.cc/paper_files/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html . Attention-based distillation produces compact pretrained transformers. This is background for the architecture family, not a complete citation for the exact sentence-embedding checkpoint. Key `wang2020minilm`.

**Reimers and Gurevych, 2019, Sentence-BERT**, EMNLP-IJCNLP. Primary https://aclanthology.org/D19-1410/ . Sentence embeddings from siamese/triplet training. Key `reimers2019sbert`.

**Reimers and Gurevych, 2020, Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation**, EMNLP. Primary https://aclanthology.org/2020.emnlp-main.365/ . Aligns multilingual sentence representations through teacher distillation. Key `reimers2020multilingual`.

Exact project checkpoint is confirmed by `configs/pretrained_rtdetr.json`: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Its primary model card https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 describes 384-dimensional mean-pooled embeddings and cites SBERT. The model is multilingual, but the project's supported instructions are English; do not claim multilingual task generalization.

## Broader manipulation context

**Zeng et al., 2021, Transporter Networks**, proceedings of CoRL 2020 (publication year 2021). Primary publisher BibTeX https://proceedings.mlr.press/v155/zeng21a.html . Spatial feature transport parameterizes visual manipulation actions; this project outputs target object poses and does not learn the motor policy. The publisher author list has eleven authors; the project website includes an additional person, so the bibliography uses publisher metadata. Key `zeng2021transporter`.

**Shridhar et al., 2022, CLIPort**, proceedings of 5th CoRL (conference held 2021). Primary publisher BibTeX https://proceedings.mlr.press/v164/shridhar22a.html . Combines semantic and spatial streams for language-conditioned manipulation. Cite as context for separating language semantics and geometry, not as an evaluated baseline. Key `shridhar2022cliport`.

**Jiang et al., 2023, VIMA**, ICML. Primary publisher BibTeX https://proceedings.mlr.press/v202/jiang23b.html . Multimodal prompts condition an autoregressive action policy; richer task/scene generalization than the current project's template-controlled evaluation. Use the published title *Robot Manipulation with Multimodal Prompts* (without “General,” which appears in the arXiv title). Key `jiang2023vima`.

## Official template provenance and use

The official ICLR 2026 Author Guide https://iclr.cc/Conferences/2026/AuthorGuide links to `https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip`. Exact upstream `.sty` and `.bst` are stored unmodified in `paper_submit/template/`, with URLs, immutable commit and SHA-256 hashes in `upstream.json`.

Use `\documentclass{article}`, `\usepackage{template/iclr2026_conference,times}`, `\iclrfinalcopy`, and `\bibliographystyle{template/iclr2026_conference}`. The template sets a misleading conference-acceptance header inside `\maketitle`; immediately afterward set `\lhead{Course Research Paper}` (or a short project header) and an appropriate footline. The source can remain byte-identical upstream. Do not use anonymous-review branding or claim ICLR acceptance. Course assignment instructions take precedence over the template's conference-specific requirements and 9-page limit. A five-page main text is compatible with retaining the template's font and dimensions.
