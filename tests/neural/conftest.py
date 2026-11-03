import pytest
torch = pytest.importorskip('torch', reason='install the neural extra for S0 tests')
torch.set_num_threads(2)


@pytest.fixture
def s0_dataset():
    from pathlib import Path
    from knolling_model.data import S0Dataset
    root = Path('artifacts/pilot_v1_3_en')
    if not root.exists():
        pytest.skip('local pilot artifact not installed')
    return S0Dataset(root, 'configs/s0_samples.json')
