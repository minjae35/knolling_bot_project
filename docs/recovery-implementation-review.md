# Neural Knolling recovery — implementation review

**상태: 구현·CPU 검증 단계. GPU 학습, optimizer update, 실험 실행, 성능 개선 입증은 하지 않았다.**

브랜치: `codex/neural-recovery-review`  
기준 커밋: `dbd68d08df936d6d470c62fab4e017c57c76ddeb`  
검토 설정: [`configs/recovery_review.json`](../configs/recovery_review.json)

## 1. 변경 목적과 보존 범위

기존 S0에서는 검출 개수 오류, confidence와 geometry의 결합, 언어별 target orientation 구분 실패가 동시에 발생했다. 복구 구현은 검출과 layout을 각각 검증한 후 연결할 수 있게 만든다.

- 유지한 목표: **BEV RGB + 자연어 → 물체별 target `(x, y, theta)`**.
- 추론 입력은 RGB와 tokenized text뿐이다. Teacher, GT 개수, GT source 속성, 정답 prefix, 명령 parser, layout repair planner를 사용하지 않는다.
- 기존 `knolling_model` 모델·runner·loss·evaluator 및 기존 Teacher는 변경하지 않았다. 새 구현은 `knolling_model/recovery/`와 별도 Teacher 모듈에 있다.
- 기존 Dataset, Checkpoint, 실험 결과를 읽기 전용 입력으로만 사용했다. 새 데이터셋·학습 checkpoint를 생성하지 않았다.
- 기존 학습 checkpoint와 새 모델은 구조가 다르다. `load_recovery()`는 기존에 pin한 R18/MiniLM 기반 가중치를 캐시에서 읽고 새 task head/layout을 초기화한다. legacy S0 checkpoint를 자동 변환하지 않는다.

## 2. 구현된 아키텍처

```text
BEV RGB ── RT-DETR-R18 + task foreground + physical heads
                    │
                    └─ xy / size / periodic angle / color·shape probabilities
                       + valid mask + source query identity
English ── frozen MiniLM ── trainable projection
                    │
                    └─ object/text Transformer ── autoregressive pose decoder
                                                 ├─ xy
                                                 ├─ angle bin + residual
                                                 ├─ move/keep
                                                 └─ selection, axis, region auxiliaries
```

| 구성 | 구현 | 검토 시 유의점 |
|---|---|---|
| 검출 | COCO class max 대신 encoder/각 decoder에 1-class foreground head | 새 head는 초기화되므로 학습 전 confidence 성능을 주장할 수 없음 |
| Detection loss | VFL 1 / box L1 5 / GIoU 2, matcher 2/5/2, encoder·decoder auxiliary, DN | geometry gradient를 유지하는 adapter이므로 native 학습의 detach 정책과는 다름 |
| Denoising | GT DN query를 normal query와 분리, self-attention mask 적용 | 모든 normal stage의 matching·physical head에서 DN 제외 |
| 물체 인터페이스 | 12개 연속 속성: xy 2 + size 2 + cos/sin(2θ) 2 + color 3 + shape 3 | raw detector token과 confidence를 layout geometry 입력에서 제거 |
| Layout | 기본 128차원, 2층 object/text encoder + 2층 causal decoder | 원래 source query 순서로 출력; target 위치로 정렬하지 않음 |
| Orientation | π-periodic 18 bins + 각 bin의 residual | 10° 간격; 수평·수직이 bin 중심. 최종 bin은 hard 선택 |
| Move/keep | 학습된 keep 결정이면 해당 source pose를 복사 | GT의 정확한 변화량 기준 label 사용. 기존 3mm/3° 평가 label과 구분 |
| 언어 | MiniLM 고정, projection·layout 학습, selection/axis/region 보조 loss | 보조 정답은 loss에만 제공. 추론에서 정답 의미 필드를 전달하지 않음 |

물체 선택 threshold와 angle bin/move 결정은 이산적이다. 연속 source 속성을 통한 RGB→target gradient 경로를 검증했으며, 이산 선택 자체가 미분 가능하다고 주장하지 않는다. Move/selection/bin은 별도 분류 loss로 학습한다.

**HF 4.57.3 관련 주의:** 설치된 `RTDetrForObjectDetectionLoss` wrapper는 auxiliary DN split과 달리 final Hungarian 입력에서는 DN을 분리하지 않는다. 새 adapter는 `RTDetrLoss`에 normal query만 전달하고 DN loss를 별도로 구성한다. 작은 실제 RT-DETR 모듈에서 normal query 출력의 DN 유무 동등성 및 matching query 수를 테스트했다. 의존성 버전은 4.57.3으로 제한했다.

## 3. 단계별 학습 연결 — 실행하지 않음

[`stages.py`](../src/knolling_model/recovery/stages.py)는 loss와 동결 설정만 구성한다. `backward`, optimizer 생성/step, checkpoint 저장, GPU 전송 또는 학습 loop는 실행하지 않는다. 향후 실제 학습 driver·resume·checkpoint 운용은 검토 후 별도로 연결해야 한다. 기존 S0 runner는 recovery 설정 schema를 거부한다.

| 단계 | 입력/학습 대상 | 구현된 분리 | 향후 제안 상한 |
|---|---|---|---|
| 1 `oracle_layout` | GT source 속성 + 실제 English, layout만 학습 | 명시적인 진단 경로. 최종 시스템 성능으로 보고 금지 | 500 updates / 10분 |
| 2 `perception` | RGB, 검출·physical heads | 원본 장면을 중복 없이 순회, native-style detection + DN | 1,000 / 45분 |
| 3 `predicted_layout` | 예측 source 속성 + English, perception 고정 | source-only matching으로 학습 label 정렬. optional measured noise | 1,000 / 30분 |
| 4 `joint` | RGB + English | 마지막 detector decoder/box/class head와 physical heads만 해제, layout 학습, 별도 detection replay 필수 | 300 / 15분 |

상한은 제안값이다. 이 구현에는 update/time 실행 loop가 없으므로 실행 상한을 집행했다고 주장하지 않는다. `configure_stage()`는 shape/gradient 진단을 위해 각 단계를 직접 선택할 수 있다. `stage_gate()`는 측정값을 검증하는 함수이며, 실제 학습의 단계 전환을 자동 승인하지 않는다.

- Layout 노출은 full/partial/no-op를 동일 횟수로 맞추면서 모든 명령을 최소 한 번 포함한다.
- Perception은 S0의 25개 unique source를 한 번씩 순회한다. Joint에는 이 별도 stream을 넣으며, 한 epoch 안에 전체 source와 전체 instruction을 모두 포함한다.
- Teacher forcing은 layout warmup 전용 API이며 eval mode에서 거부한다. 기본 경로는 free rollout이다.
- Stage 3/4의 training matching은 정답 source와 대응되는 query를 confidence 아래에서도 고를 수 있다. **이 결과는 조건부 진단**이다. 추론에서는 예측 foreground만으로 물체를 선택하므로 별도 count/coverage gate를 반드시 통과해야 한다.
- Noise는 caller가 명시한 측정 표준편차와 generator를 요구한다. 임의의 noise 값이나 semantic confusion model을 기본값으로 넣지 않았다.
- 제안 gate: oracle target P95 ≤1mm/2°, move·pure-axis 통과 → perception count/coverage 100%, FP/FN 0, source P95 ≤1mm/2° → strict legacy S0 및 semantic/counterfactual 통과 → joint 전체 회귀 없음.
- 실제 gate 결과는 아직 없다. 단위 테스트의 synthetic 통과는 이 gate의 실험 통과가 아니다.

## 4. 데이터와 Teacher 후보

[`data.py`](../src/knolling_model/recovery/data.py)는 기존 데이터를 감싸 supervision만 확장한다. family split 중복 검증, axis/region/gap/order/group 한 요인 contrast, observable selection contrast 생성 함수를 추가했다. 반환값은 메모리 내 instruction이며, feasibility와 teacher target 검증 후 새 버전 데이터로 작성해야 한다. 기존 130개 record를 재작성하지 않았다.

[`teacher_continuous.py`](../src/knolling_dataset/teacher_continuous.py)는 `continuous_envelope_v0_review`라는 **offline 전용 후보**다. 기존 정책 registry/default에 등록하지 않았다.

- Balanced row template의 이동 중심을 연속 좌표로 선택하므로 기존 5mm 이동 격자를 사용하지 않는다.
- Locked object는 정확히 보존하며, 모든 결과를 기존 geometry/semantic validator로 확인한다.
- Bounding-envelope obstacle 제약은 보수적이다. 후보 소진은 `generation_failure`로 반환하며 실제 물리적 infeasible이라고 단정하지 않는다.
- Template 변경, 영역 경계, 장애물 양쪽 선택, 동률 등에서 불연속은 여전히 가능하다. 작은 단일 perturbation fixture 통과가 전체 안정성을 보장하지 않는다.
- 아직 64–128개 contrast 생성, 대규모 geometric-only corpus 생성, render, perturbation sweep을 수행하지 않았다.

## 5. 평가·검증

재현 명령 — 기존 캐시를 사용하며 CPU만 노출:

```bash
HF_HOME=/workspace/cache/huggingface TORCH_HOME=/workspace/cache/torch \
  /venv/knolling/bin/python scripts/verify_recovery_cpu.py --cached-model
```

Script는 Python AST 문법 검사, `git diff --check`, 전체 pytest를 수행한다. 테스트 subprocess에 `CUDA_VISIBLE_DEVICES=''`, Hub offline 설정을 강제한다. 새 모델 테스트에는 optimizer step이 없다. 기존 회귀 테스트의 tiny optimizer/checkpoint fixture는 임시 디렉터리에서만 동작한다.

검증 항목:

- Source meter/image 좌표, 회전 envelope box, ID join 및 padding, source-only detector label.
- DN batch 크기 2와 서로 다른 object 수, normal query와 GT DN의 격리, native eval output 동등성.
- 작은 실제 RT-DETR 및 캐시된 실제 R18/MiniLM의 CPU shape·finite backward.
- RGB→target, geometry/text projection gradient; frozen text/perception에는 gradient 없음.
- π 경계·수평/수직 bin, keep copy, autoregressive causality, free-rollout prefix 재현, padding 격리.
- 단계별 parameter freeze, 별도 replay 요구, 전체 source/instruction 노출, teacher-forced eval 거부.
- 모델이 capacity 8을 넘는 물체를 검출하면 무단 자르지 않고 오류로 표시. 평가에서는 해당 샘플을 count 실패로 남기며 pose 통계에서 성공 샘플처럼 포함하지 않음.
- strict S0 evaluator를 단일 최종 target stage로 재사용하며 기존 threshold 0.5와 denominator를 유지.

검증 기록 (2026-11-08 UTC):

- Python AST 문법 검사: **92개 파일 통과**. `git diff --check` 통과.
- 전체 pytest: **175 passed, 11 skipped**, 37.08초. 새 recovery/Teacher 테스트 24개가 포함되며 실제 cached R18/MiniLM CPU 검사도 통과했다.
- 생략 11개: CUDA 전용 1개, 별도 opt-in legacy pretrained 검사 2개, 렌더링·migration·전체 pilot deliverable 검사 8개. 이번 범위에서 CUDA/렌더링을 켜지 않았다.
- 새 모델의 Unit/Shape/Gradient 검증에는 optimizer update가 없으며, 실제 R18/MiniLM에서도 forward/backward만 수행했다.
- `artifacts/`의 기존 **2,105개 파일**을 작업 전후의 경로·크기·수정 시각으로 비교: 변경·삭제·추가 0개. 데이터 adapter의 기존 manifest/annotation/RGB hash 검증도 통과했다. 전체 checkpoint 바이트를 다시 해시한 검증은 아니다.
- 종료 확인 시 GPU compute process 없음. GPU 학습·본격 실험·성능 측정 미실행.

## 6. Work가 독립적으로 검토할 항목

1. **GT 유출:** DN mask와 normal/DN split, final/aux matching, `forward()` 인자, oracle/teacher-forced 경로가 공식 평가에 섞이지 않는지.
2. **좌표·대응:** y 반전·정규화·π 주기, native box matching의 identity 정확도, source query 순서가 바뀔 때 AR 학습의 안정성.
3. **Loss 설계:** foreground VFL 초기화, physical loss 규모, exact move label과 최종 허용오차의 차이, angle bin residual 경계, hard keep/bin 선택의 gradient 한계.
4. **단계별 데이터:** source 중복 제거, full/partial/no-op 노출, stage 3의 조건부 GT matching과 실제 count/coverage 사이 차이, joint replay 및 해제 범위.
5. **Teacher 후보:** 보수적인 envelope 제약, template/장애물 경계 불연속, no-op 판정, 기존 Teacher와 비교할 별도 버전·평가 정책.
6. **승인 후 실행 계획:** Stage 1 실패 시 언어/layout부터 수정; Stage 2 실패 시 검출에 한정. 별도 output 디렉터리, schema가 있는 checkpoint/resume, update/time cap, gate 실패 중단·회귀 시 rollback을 학습 driver에 연결한 뒤 실행.

아직 입증하지 않은 것은 **수렴, 실제 정확도, collision-free 성공률, counterfactual 언어 이해, 일반화 및 GPU 성능**이다. CPU gradient가 존재하는 것과 시스템이 학습에 성공하는 것은 별개의 결과다.

## 7. 설계 참고

- [RT-DETR criterion](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py): detection matching/VFL/box/auxiliary/DN 구성 참고.
- [Knolling Bot](https://arxiv.org/abs/2310.04566), [StructFormer](https://arxiv.org/abs/2110.10189): object 중심 표현과 autoregressive arrangement 참고. 입력·언어·데이터 조건이 이 프로젝트와 같다는 주장은 하지 않는다.
- [MultiBin](https://arxiv.org/abs/1612.00496): orientation의 분류+residual 아이디어 참고. 현재 구현은 π-periodic non-overlapping bin으로, 원 논문의 완전한 재현은 아니다.
