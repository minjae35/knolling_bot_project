# Neural Knolling recovery — 수정 후 재검토 자료

**상태: 코드·CPU/CUDA 검증 완료, 실제 모델 학습 및 성능 실험 미실행. Work 재검토 대기.**

- 브랜치: `feat/neural-recovery`; 수정 전 기준: `9b141c3`.
- 이 문서는 이전 구현 보고서를 갱신한다. 이전 보고서는 Git의 `9b141c3`에서 확인할 수 있다.
- 실행 설정: [`recovery_execution.json`](../configs/recovery_execution.json). 기본값은 `review_only`, `allow_training=false`다.
- 모든 Python 검증은 Conda `knolling`(`/venv/knolling`)에서 수행했다. 추가 패키지를 설치하지 않았다.

## 1. 검토 지적과 실제 수정

| 지적 / 확인한 원인 | 수정 | 확인한 근거 |
|---|---|---|
| Arrangement 학습은 전체 query에서 Hungarian matching하고 추론은 foreground ≥0.5를 선택 | 공통 `select_queries()`로 먼저 같은 query를 선택하고, 그 안에서 source 정답만 대응. 개수·capacity·10mm 대응 조건 실패 시 `SourceSelectionError` | confidence 경계에서 학습 `[0,2]` / 추론 `[0,1]`로 갈리던 회귀 fixture; 선택 query 동일성 및 CUDA train/inference 출력 비교 |
| 평가 출력과 gate 입력 schema가 달라 실제 측정으로 통과 불가; 단일 target을 3-stage처럼 보고 | `recovery_metrics_v2`와 실제 evaluator를 직접 연결. oracle/perception/predicted 경로 분리, 예상 sample ID 전부 존재해야 complete | 실제 evaluator 출력으로 4개 gate 검사, 누락 평가·무시된 axis 명령·중복 검출 실패 확인. fixture 통과는 모델 성능이 아님 |
| 학습 단계 함수만 있고 실행·중단·checkpoint 운용 없음 | 1회 호출당 1단계 실행 driver, 이전 gate 통과 요구, update/time 제한, checkpoint/resume/rollback 구현 | scalar fixture에서 연속/재개 실행의 가중치·Adam 상태·cursor 일치, 단계 건너뛰기 거부, nonfinite/selection/회귀 rollback |
| Counterfactual은 instruction 생성 함수만 존재; 새 Teacher/로더 연결 없음 | 검증된 64개 target record 생성, 원본 32개와 `RecoverySuite`로 연결. Teacher replay·hash·family/split·영어 의미 검사 | 32쌍 모두 양방향 명령 구분 및 geometry 검증, 64개 Teacher target 재현, 전체 bundle 재생성 SHA 일치 |

구현 위치: [`selection.py`](../src/knolling_model/recovery/selection.py), [`evaluation.py`](../src/knolling_model/recovery/evaluation.py), [`stages.py`](../src/knolling_model/recovery/stages.py), [`driver.py`](../src/knolling_model/recovery/driver.py), [`counterfactuals.py`](../src/knolling_model/recovery/counterfactuals.py).

## 2. 유지한 아키텍처와 GT 경계

```text
BEV RGB → RT-DETR-R18 foreground / physical heads
          → 추론 foreground query 선택
          → xy, size, periodic angle, color/shape probabilities, query identity
English → frozen MiniLM → trainable projection
          → object/text Transformer → autoregressive target pose + move/keep
```

- 추론은 RGB와 tokenized English만 받는다. GT 개수·속성·target prefix·명령 parser·Teacher·rule-based 배치 planner를 전달하지 않는다.
- Oracle 단계만 GT source를 받는 명시적 진단이다. Predicted/joint에서는 **query 선택 후** GT를 loss label에 대응한다. 부족한 검출을 GT로 보충하지 않는다.
- Joint에서는 연속 source 속성을 통한 layout → perception gradient를 유지한다. Foreground threshold, move 결정, orientation bin의 이산 선택 자체가 미분 가능하다는 주장은 하지 않는다.
- 기존 DN 분리·mask, 1-class foreground, π-periodic bin/residual, learned keep, 별도 unique-source detection replay를 유지했다.
- 이번 수정에서 기존 legacy evaluator의 기본 3-stage 동작은 유지하고, 선택적인 `stage_count`만 추가했다. Recovery target은 1개, perception-only 보고서는 layout 결과를 미측정으로 표시한다.

## 3. 실제 평가와 단계 통과 기준

모든 gate는 올바른 schema/단계, free rollout, pin한 전체 sample ID 일치를 요구한다. 물체를 누락하거나 capacity를 초과한 샘플은 실패로 남긴다.

| 단계 | 학습 대상 / 데이터 | 통과 기준 | 실행 상한 |
|---|---|---|---|
| `oracle_layout` | GT source + English, layout | target P95 ≤1mm/2°, move 100%, pure-axis 100% | 500 updates / 600초 |
| `perception` | RGB, 25개 unique source | count/coverage 100%, FP/FN 0, source 위치·크기 P95 ≤1mm, 각도 ≤2°, color/shape 100% | 1,000 / 2,700초 |
| `predicted_layout` | 예측 source + English, perception 고정 | source 기준 + legacy S0 gate + 전체 task/pure-axis 100%, mode별 회귀 없음 | 1,000 / 1,800초 |
| `joint` | layout 및 detector 마지막 decoder/heads, 별도 detection replay | 위 기준 + 단계 시작 reference 대비 전체 회귀 없음 | 300 / 900초 |

- FP/FN은 source 중심거리 ≤10mm 및 footprint IoU ≥0.5의 독립 대응으로 센다. 단순 검출 개수 차이로 서로 상쇄하지 않는다.
- Pure-axis는 같은 RGB에서 axis만 바꾼 검증된 pair에 대해 **두 자기 명령 성공 + 두 반대 명령 실패**를 요구한다. pair가 없거나 반쪽만 평가되면 통과하지 못한다.
- Color/shape/move 정확도의 분모는 전체 GT 물체다. 미검출을 분모에서 제거하지 않는다.
- 전체 회귀 비교는 동일 sample set의 count/coverage/FP/FN, source·target P95, task/axis/move/color/shape 및 mode별 task success를 포함한다.
- Oracle target P95 기준은 모든 물체의 최대 오차 기준이 아니다. Predicted/joint 단계는 추가로 전체 semantic/geometry task success를 요구한다.
- **실제 모델이 이 gate를 통과했다는 결과는 아직 없다.**

## 4. Dataset / Teacher 검증과 한계

새 bundle: `artifacts/recovery_v2_controls/counterfactuals.json`.

- **64 records / 32 pairs:** pure-axis 21쌍, region 5쌍, selection 6쌍.
- 원본 S0 32개를 포함한 suite는 **96개**, 원래 **25 source families**를 공유한다. RGB를 다시 렌더링하지 않았다.
- 기존 split을 그대로 상속한다. 이 suite는 Train의 S0 복구·명령 구분용이며 **held-out 일반화 평가 데이터가 아니다**.
- 부모 manifest, source, source RGB SHA, family/split, pair 구성, 영어와 구조화 명령의 일치, geometry/semantic 유효성을 확인한다. 기본 로딩 시 새 Teacher를 재실행하여 target과 비교한다.
- 기존 Teacher와 새 Teacher가 **동일 source/RGB와 동일 문장에 다른 target**을 붙이는 경우를 발견해 생성 시 제외하고 로더에서도 거부한다. 기존 label은 그대로 유지한다.
- 새 `continuous_envelope_v0_review` Teacher는 offline 전용이며 기존 default Teacher를 바꾸지 않았다. 새 contrast target에만 적용한다.
- 연속 위치 투영, locked object 보존, no-op/full/partial 유효성, 10μm 국소 perturbation, 20개 obstacle-union fixture의 격자 참조 대비 거리 검사를 통과했다.
- Template 전환·장애물 양쪽 동률에서 불연속이 남는다. 보수적인 envelope 때문에 가능한 문제도 generation failure로 거부할 수 있다. 대규모 안정성·일반화를 입증하지 않았다.

Bundle SHA-256:
`2110de927ab86fc3b0b6e2a7f8a768a61d71e7ef4da23d13136f58217facccac`

Suite fingerprint:
`e515a4694af4a57bc803b0a40e86665e96f596bd30aa8f01de834b485ed7db2b`

Bundle은 로컬 artifact이며 Git에 포함하지 않는다. 같은 부모 dataset이 있는 새 checkout에서 다음 명령으로 재생성한다. 이미 있는 출력 폴더는 덮어쓰지 않는다.

```bash
conda run --no-capture-output -n knolling python -m knolling_model.recovery.counterfactuals \
  --output artifacts/recovery_v2_controls --pairs 32
```

## 5. Driver 운용 계약 — 아직 실행 승인하지 않음

- CLI: `python -m knolling_model.recovery.driver`. 반드시 `knolling` 환경에서 사용한다.
- `--stage`, 새로운 `--output`, `--approval`이 필요하다. 다음 단계는 `--parent-checkpoint`, 동일 단계 재개는 `--resume`을 사용하며 두 옵션은 동시에 사용할 수 없다.
- 기본 설정은 model/CUDA/output 생성 전에 실행을 거부한다. 향후 실제 실행에는 검토된 별도 설정의 `execution=approved_training`, `allow_training=true`와 review receipt가 필요하다.
- Receipt 필드: `approved`, `reviewer`, 정확한 `commit`, `config_sha256`(`schema.digest(config)`), `dataset_fingerprint`, 허용 `stages`; CUDA 학습에는 `gpu_training=true` 추가. 이는 실행 실수를 막는 검토 기록이지 서명된 보안 인증은 아니다.
- 깨끗한 checkout과 code/config/data가 일치해야 한다. 설정 및 receipt도 checkpoint에 남긴다. 이번 작업에서는 실제 학습 승인 receipt를 만들지 않았다.
- Checkpoint는 model, optimizer, Python/NumPy/Torch/CUDA RNG, update/epoch/batch cursor, 누적 시간, baseline/완료 gate, stage-entry anchor hash를 저장한다. 기존 파일을 원자적으로 덮어쓰지 않는 방식으로 생성한다.
- SIGINT/SIGTERM은 step 경계에서 pause한다. Forward/backward 도중 중단되어 update하지 않은 batch는 RNG도 복원하여 재개 때 다시 계산한다. 강제 종료는 마지막 durable checkpoint부터 재개한다.
- 매 실행/재개마다 새 output 디렉터리를 요구한다. `running`/`paused`만 재개 가능하다. `budget_exhausted`/`rolled_back`은 자동 재개하지 않는다.
- Nonfinite loss/gradient/parameter, selection 실패, joint 회귀는 해당 단계 시작 anchor로 rollback하고 오류 checkpoint를 남긴다. Rollback checkpoint의 optimizer moments는 거부된 경로의 기록이므로 재개에 사용하지 않는다.
- 이전 gate가 통과한 checkpoint 없이 다음 단계로 진입하지 않는다. 기본 `stop_on_initial_pass=true`는 처음부터 통과한 단계를 update 없이 종료한다.
- 시간 제한은 optimizer update를 막는 cooperative 제한이다. 이미 진행 중인 kernel·평가·checkpoint 저장을 강제로 끊지는 않으므로 프로세스 종료 시각은 상한을 넘을 수 있다.
- 기본 경로는 free rollout이다. Source-noise/teacher-forcing 보조 API는 유지하지만 측정된 noise 정책이 없으므로 driver에서 자동 활성화하지 않는다.

## 6. 검증 결과

CPU 재현:

```bash
HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch \
  conda run --no-capture-output -n knolling python scripts/verify_recovery_cpu.py --cached-model
```

CUDA 재현 — optimizer 생성/step 없이 forward/backward만 수행:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RUN_RECOVERY_CUDA_TESTS=1 \
  conda run --no-capture-output -n knolling python -m pytest tests/neural/test_recovery_cuda.py -q -s
```

- AST: **100 Python files**; `git diff --check` 통과.
- 전체 CPU pytest: **187 passed, 12 skipped**. 실제 cached R18/MiniLM CPU shape/gradient 검사 포함.
- Skip: CUDA 2개(이 중 recovery CUDA 검사는 아래에서 별도 실행), opt-in legacy pretrained 2개, 렌더링·migration·전체 pilot 검사 8개. 전체 Blender 재생성은 하지 않았다.
- Driver 단위 테스트 **6개**: scalar fixture만 최적화했다. 실제 Knolling 모델 학습과 구분한다.
- CUDA: **1 passed**, RTX 4090, Torch `2.9.1+cu128`, CUDA `12.8`. Perception/DN, oracle layout, predicted layout, joint의 finite forward/backward, layout→perception gradient, joint detection replay backward 확인.
- Predicted/joint CUDA 경로는 미학습 detector의 confidence만 통제한 synthetic 선택 fixture를 사용했다. 미학습 모델 그대로는 `SourceSelectionError`를 내는 것도 확인했다. 이 검사는 정확도를 평가하지 않는다.
- GPU peak allocated **1,146.4 MiB**, reserved **1,240.0 MiB**. Batch 1의 검증값이며 optimizer state/실제 학습 peak의 추정치가 아니다.
- 실제 모델 **optimizer update 0회**. 검사 전후 전체 state_dict SHA 비교로 가중치·버퍼 불변 확인. 종료 시 GPU compute process 없음.
- 기존 `artifacts/` **2,105개 파일**의 경로·크기·mtime 비교: 변경/삭제 0개. 추가된 파일은 새 bundle과 SHA 파일 2개뿐이다. 기존 checkpoint 전체 바이트를 재해시한 검증은 아니다.

## 7. Work 재검토 우선순위

1. **Query/GT 경계:** 선택 정책이 train/inference에서 동일한지, source matching이 label join에만 쓰이는지, strict abort가 기대한 stage 3/4 운용과 맞는지.
2. **평가/gate:** complete sample set, oracle와 predicted 경로 구분, physical FP/FN, pure-axis 양방향 조건, source-only 보고서, regression 분모·허용오차.
3. **실행:** checkpoint 계약, stage 순서, data cursor/RNG 재개, 누적 시간 제한, stage-entry rollback 및 비재개 상태. 검토 후 승인할 설정·상한을 확정할 것.
4. **Teacher/data:** 새 label의 provenance/재현, 기존 label과의 충돌 차단, conservative envelope 및 경계 불연속. 기존 25 family 기반인 점을 감안하고 일반화 자료와 구분할 것.
5. **남은 설계 위험:** detector query 순서 변화에 대한 AR 안정성, hard move/bin 선택의 gradient 한계, loss 간 규모, 높은 source gate의 달성 가능성. 이번 테스트로 수렴·정확도·실제 knolling 성공률을 입증하지 않았다.

이 수정은 **재검토 가능한 구현 상태**를 만든 것이다. 실제 GPU 학습의 시작 여부는 Work 재검토 이후 결정한다.
