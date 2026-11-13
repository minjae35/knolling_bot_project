# neural_sequential_xy_v1 참고 스냅샷

이 디렉터리는 `experiment/neural-sequential-phase1` 커밋 `4019217954eaa8b9d9a322ad2bc48d278e0bca2c`의 `neural_sequential_xy_v1`를 그대로 얼려 둔 참고 사본이다.

현재 end-to-end 아키텍처가 아니다. `current.py`, 기본 모델, 그 밖의 라이브 진입점에 연결하지 않는다. 추론에서 GT, 규칙 플래너, 기하 솔버를 쓰면 안 된다.

실패한 형제 실험은 일부러 빼 두었다. direct-centre, relational row/slot, StructDiffusion, single-row decoder, gap auxiliary loss, teacher-prefix X-offset loss, size-conditioned offset head, Set Network, pairwise XY는 이 스냅샷에 없다.

## 들어 있는 파일

- `neural_sequential.py`: 위 커밋의 `src/knolling_model/experiments/neural_sequential.py`와 내용이 같다. `ScenePoseDecoder`가 이 파일 안에 있다.
- `test_neural_sequential.py`: 같은 커밋의 단위 테스트 사본이다. 실행 스위트로 쓰지 않는다. `test_phase1_single_row`의 `Reference`, `example`, `device`를 import하고, 라이브 경로 `knolling_model.experiments.neural_sequential`를 import한다. 그 의존 파일은 복사하지 않았다.
- `docs/neural-sequential-phase1.md`
- `configs/neural_sequential_phase1.json`

## 넣지 않은 파일

- `scripts/run_neural_sequential.py`는 `knolling_model.experiments.direct_center`, `knolling_model.experiments.phase1_evaluation`, `run_phase1_single_row_generalization`, `run_phase1_single_row_overfit`, `verify_phase1_single_row`를 import해서 복사하지 않았다.
- `tests/neural/test_neural_sequential_driver.py`는 그 러너를 import해서 복사하지 않았다.
- `offset_supervision.py`, `size_conditioned_offset.py`, `horizontal_gap.py`와 미추적 실험 러너는 거절된 실험이라 커밋하지 않았다.

## 이 모듈이 기대는 import

`neural_sequential.py`는 실패한 실험 모듈을 직접 import한다. 그 파일은 `src/`에 다시 넣지 않았다.

- `.generalization.new_model`
- `.phase1_single_row.position_loss` (`position_loss`는 이 모듈 이름으로도 다시 export된다)

그 밖에는 당시에도 쓰이던 `knolling_dataset.builder.file_hash`와 `knolling_model.recovery.storage.load_artifact`만 import한다.
