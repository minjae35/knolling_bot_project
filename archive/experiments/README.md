# Historical experiments

These scripts are retained for provenance, not the current development entry
point. Use the repository root README and `configs/current_models.json`.
Paths in old experiment reports describe their execution at the original Git
commit. Run a historical script only with an explicit new experiment request;
moving it here is not approval to replay it.

AR/v3 constructors, old checkpoint readers and their tests remain in
`src/knolling_model/recovery` for backwards compatibility. They share bin,
contract, evaluation and loss primitives with the accepted v4 model, so removing
them would break historical checkpoint restoration. Failed Pairwise code is
not part of main. No dataset, model binary or recorded experiment was deleted.
