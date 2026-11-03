# Knolling Bot Technical Specification v1.3

작성일: 2026-11-03, America/New_York

기준 구현: `69cf56549efa7f178622e6eadcc6d0eeb6b749d6`

문서 상태: 최종 통합 명세. 아래 1절의 완료 기록, 7절의 비교 및 18절의 구현 체크리스트는 `69cf565` 시점의 설계 기준을 보존한 것이다. 이후 Dataset v1.3 구현·Migration은 `400d9c4`까지 완료됐으며, 결과는 [v1.3 Pilot 보고서](v1.3-pilot-report.md)를 따른다. 해당 보고서의 사람 시각 검토 pending 기록은 당시 상태로 보존한다. 이번 문서 확정은 위치 기반 참조의 No-op 재적용, Legacy 방향 호환성, Partial 평가 기준을 명확히 하며, 이 보정 자체에 대한 코드 변경이나 새 검증 완료를 주장하지 않는다.

2026-11-03 구현 상태 갱신: 사용자가 최종 Dataset 품질 검토 완료를 확인하고 S0 구현을 요청했다. 기존 Architecture·Learning Strategy를 유지한 Adapter, Model, Matching/Loss, Evaluator, Launcher의 구현 및 로컬 검증은 [S0 구현·실행 계약](s0-implementation.md)에 기록한다. Dataset·Teacher·과거 승인 Sidecar는 변경하지 않았다. 학습은 실행하지 않았으며, 실제 시작은 이번 구현 검토 이후다. 아래 설계 기준·과거 체크리스트와 실제 완료 상태를 구분한다.

목표는 BEV RGB 이미지와 자연어 지시에서 물체별 목표 배치까지 하나의 Neural Network로 예측하는 Knolling System을 완성하는 것이다. 기존 M0–M8, 검증된 Annotation 계약, 100 Scene Pilot을 보존한다. Teacher는 제한된 후보 중 균형 있는 배치를 선택하도록 수정하고, 초기 Neural 학습은 Detection + Target Loss로 시작한다. Novelty보다 정돈 성공률, 구현 가능성, 실패 원인의 식별을 우선한다.

이 문서는 Dataset, Teacher, Model, Loss, Evaluation, Simulation의 계약을 함께 정의한다. **현재 완료 사실**, **다음 구현에서 필수인 변경**, **조건 충족 후 진행할 확장**을 구분한다. Hyperparameter와 성능 기준은 실행을 위한 초기 결정이며, 실험적으로 확인된 최적값이나 달성 성능이 아니다.

## 1. 현재 상태와 보존 대상

### 1.1 완료된 작업

| 항목 | 확인된 상태 |
|---|---|
| Dataset Generator | M0–M8 구현 완료 |
| 테스트 | 기존 완료 보고서에 전체 39개 통과 기록 |
| Git | 12개 Commit; 로컬 HEAD와 로컬 `origin/main` 기록이 `69cf565`로 일치; main Push 완료는 사용자 보고와 기존 보고서 기준 |
| 데이터 | 100 Scene Family, 130 Sample, 210 Render View |
| Primary | Full 50, Partial 30, No-op 20 |
| No-op | 반복 지시 14, 명시적 Keep-all 6 |
| Counterfactual | 30 Family에 동일 Source 이미지의 다른 지시·Target |
| 채택률 | Primary 100/101 = 99.01%; 추가 지시 30/30 |
| 자동 감사 | 기하, Identity, Instruction, Mask, 좌표, Split 및 파일 정합성 오류 0건 |
| 사람 검토 | 130 Sample 전수 검토 미완료 |
| Neural / 로봇 | 구현·학습·로봇 실행 완료 근거 없음; 다음 단계 |

근거는 [Pilot 보고서](pilot-report.md), [현재 데이터 계약](dataset-contract.md), [구현 결정 기록](implementation-decisions.md)이다. 39개 테스트와 자동 감사는 기존 실행 결과이며, 이번 문서 작성에서 다시 실행하지 않았다. 생성 채택률은 Neural 성공률이나 미적 품질 점수가 아니다.

보존할 기준 데이터는 `artifacts/pilot_v1_2/`이다. Manifest SHA-256은 `b8276caeffb0287eb738888c892d11ac508913cee514873564f5785d9bc7da27`이다. 현재 provenance의 Annotation Revision `8cda9be`와 Render Revision `47ca812`는 최종 검사 Commit `69cf565`와 역할이 다르므로 덮어쓰지 않는다.

### 1.2 이번 Revision의 적용 원칙

1. 기존 코드·데이터를 폐기하거나 원본 위에 재생성하지 않는다.
2. v1.2 데이터는 유효한 Legacy Teacher 결과와 회귀 검사용으로 보존한다.
3. v1.3 후보 선택이 바뀌더라도 기존 기하·Instruction 계약을 통과한 v1.2 Target이 오류가 되는 것은 아니다.
4. 다만 동일 입력에 서로 다른 Target 정책을 섞어 학습하지 않는다. 학습 Manifest는 Teacher 정책 하나를 명시한다.
5. 이번 산출물은 문서뿐이다. 이후 코드 수정, 데이터 갱신, 학습 실행은 별도 구현 작업이다.

## 2. 목표와 지원 범위

### 2.1 성공의 정의

성공은 모든 관측 물체를 빠짐없이 대응시키고, 예측 목표 배치가 충돌·최소 간격·경계·자연어 지시·보존 조건을 함께 만족하는 것이다. Teacher 좌표와 완전히 동일해야만 성공인 것은 아니다. Target 오차는 모방 정확도 지표이며, Task Success와 따로 보고한다.

최종 배치를 별도 Rule-based Planner가 수정하거나 결정하지 않는다. Heuristic Teacher는 학습 데이터 생성과 Baseline에만 사용한다. 실행 전 Safety Validator는 실행 허용/거부만 할 수 있고 좌표를 고쳐서는 안 된다.

### 2.2 v1 지원 조건

| 항목 | 계약 |
|---|---|
| 작업대 | 0.600 × 0.400 m, 고정 보정된 BEV |
| 입력 이미지 | RGB, H=384, W=576 |
| 물체 수 | 2–8; 초기 Neural Pilot은 2–4부터 |
| 형태 | box, chamfer, capsule; 평면에 안정적으로 놓인 강체 |
| 크기 | 긴 변 a=40–80 mm, 짧은 변 b=25–50 mm, a/b≥1.3 |
| 높이 | 5–20 mm; 렌더·물리 Metadata이며 초기 NN 예측 대상 아님 |
| 치수 표본 | 1 mm 격자 |
| 색상 | red, green, blue |
| 초기 방향 | 임의 평면 방향, π 주기 |
| 새 정돈 방향 | 긴 축 수평 0 또는 수직 π/2 |
| 지시 | 색상 그룹, 외접 면적 오름/내림차순, 간격, 전체/좌/우 영역, 부분 수정, No-op |
| 요청 간격 | 15/20/30 mm |
| 제외 | 가림·적층, 방향 구분 불가능한 원/정사각형, 시각적으로 구별 불가능한 완전 중복 인스턴스, 자유형 Packing, 임의 대화 기억 |

Partial에서 고정 물체와 명시적 Keep-all에서는 기존 방향을 그대로 유지한다. 따라서 모든 Target이 0 또는 π/2여야 하는 것은 아니다. 수직 방향 지시는 물체 긴 축의 방향이며, Row를 Column 배치로 바꾸라는 뜻이 아니다.

자연어는 현재 관측으로 대상을 특정할 수 있는 **영어 전용 제한 문법**을 사용한다. 색상 또는 명확한 “the leftmost object”를 참조할 수 있다. 숨은 Object ID, 과거 대화에만 있는 지시, 해석 기준이 없는 “a little more”, “like before”는 초기 지원 범위 밖이다.

### 2.3 English-only v1 언어 정책

초기 시스템의 유일한 지원 언어는 영어다. 활성 Train/Validation/Development/Test의 모든 Instruction은 `language="en"`이어야 하며, 사용자 예시·QC·언어 반응 평가도 영어를 기본으로 한다. 한국어 및 다국어 지원은 선택적인 후속 작업이며 초기 성능 목표·학습 분포에 포함하지 않는다. 이 명세서의 설명 언어와 시스템이 지원하는 입력 언어는 별개다.

기존 `pilot_v1_3`의 한국어 73개/영어 57개는 Legacy 이력으로 보존한다. 활성 데이터는 별도 `artifacts/pilot_v1_3_en/`의 영어 130개다. 구조화된 지시 필드와 Source 관측으로 영어 문장을 결정적으로 만들고, 자유 번역이나 Teacher 재실행으로 의미·좌표를 바꾸지 않는다. 기존 영어 문장도 같은 간결한 계약으로 정리한다. 위치 참조를 관측 가능하게 표현할 수 없는 선택 집합은 숨은 ID로 대체하지 않고 변환 실패로 보고한다.

긴 축 방향, 선택/고정 대상, 색상별 순수·연속 Row, 외접 면적 순서, 수평 간격, Row 띠의 수직 간격, 색상 간 2g, 영역 및 Keep 의미를 빠짐없이 유지한다. Keep-all 예시는 “Keep every object at its current position and orientation.”이다. 문장을 간결하게 하기 위해 의미를 생략하거나 이미 정돈된 No-op 지시를 임의로 Keep-all로 바꾸지 않는다.

최종 품질 검토에서는 현재 제한 문법을 S0에 유지한다. 짧은 사용자 표현의 의미를 보존하기 위한 기존 기본 규칙은 [English arrangement conventions](english-instruction-defaults.md)에 정리한다. 해당 문서의 짧은 예시는 후속 문법 제안이며 현재 Annotation·지원 문법을 바꾸지 않는다. Slot 값이 다른 81개 문자열을 자연스러운 영어 Paraphrase 81종으로 해석하지 않는다. S0는 고정 문법의 과적합 진단이며 자연어 일반화 검증이 아니다.

## 3. 좌표와 기하 계약

원점은 작업대 좌하단, +x는 오른쪽, +y는 위쪽, θ는 +x에서 반시계 방향이다. 물체의 긴 축은 π 주기이며 표준 저장 범위는 [-π/2, π/2)이다. π/2와 -π/2는 같은 방향이다.

**Legacy 방향 호환성:** 기존 v1.2 Source/Target에는 +π/2가 저장되어 있을 수 있으며, 이를 범위 밖이라는 이유만으로 무효 처리하지 않는다. 새로 계산하는 v1.3 방향은 표준 범위로 정규화하여 수직을 −π/2로 저장하되, 기존 Source와 고정 물체·No-op Target은 원래 수치 표현까지 정확히 복사한다. 기존 JSON·이미지·Hash를 정규화 목적으로 덮어쓰지 않는다. 방향 오차·move_label·기하 비교 및 Migration의 Target 변경 여부는 π 주기로 계산하여 +π/2와 −π/2를 동등하게 취급한다. 표현만 달라졌다는 이유로 이동이나 재렌더가 필요한 Target 변경으로 판정하지 않는다. 이는 Dataset의 고정 Pose 정확 복사 계약을 완화하는 것이 아니다.

\[
u=x/0.6,\quad v=y/0.4,\qquad
x=0.6u,\quad y=0.4v.
\]

Pixel 중심은 좌상단 (0,0) 기준이다.

\[
x_{px}=576u-0.5,\quad y_{px}=384(1-v)-0.5.
\]

`source_envelope_xyxy_uv`는 작업대 좌하단 좌표계이고 이미지 y-down Box가 아니다. Mask Bounding Box는 이미지 좌표이며 최대 x/y가 exclusive이다.

GT, Teacher, Validator의 길이 단위는 meter, 각도는 radian이다. NN 중심 출력만 uv로 표현하며, 회귀 Loss의 물리적 정규화 및 기하 검사 전에 meter로 복원한다. NN 크기는 고유 a,b이며 이미지 AABB의 폭·높이가 아니다.

목표 방향에서 투영 폭·높이는 다음과 같다.

\[
w_i=a_i|\cos\theta_i|+b_i|\sin\theta_i|,\quad
h_i=a_i|\sin\theta_i|+b_i|\cos\theta_i|.
\]

모든 안전 조건은 물체의 회전 외접 직사각형 Envelope 기준이다. 실제 Mesh Footprint는 렌더 검증·물리 시뮬레이션용으로 별도 보관하며, Envelope 밖으로 나갈 수 없다. Capsule의 작은 실제 면적으로 안전 조건을 완화하지 않는다.

| 검사 | 값 |
|---|---:|
| Full Source 최소 간격 | 5 mm |
| Source 최소 경계 여유 | 10 mm |
| 현재 Partial·Keep-all 생성기의 Source 여유 | 간격·경계 12 mm, 보존 |
| 모든 유효 Target의 최소 간격·경계 여유 | 각각 10 mm |
| 새 Teacher 선택 물체의 작업 영역 | 아래 inset 영역에 Envelope 전체 포함 |
| 수치 기하 허용오차 | 0.1 mm |
| 허용하지 않는 겹침 면적 | 1e-12 m² 초과 |

영역은 (left,bottom,right,top) meter 순서다: `full=(.02,.02,.58,.38)`, `left=(.02,.02,.28,.38)`, `right=(.32,.02,.58,.38)`.

## 4. Annotation과 버전 호환성

### 4.1 기존 필드 유지

파일 형식은 JSON/JSONL과 PNG를 유지한다. **기술 명세 v1.3과 `schema_version="1.2"`는 별개**다. 데이터 구조와 필드 의미를 바꾸지 않으므로 Schema를 일괄 1.3으로 바꾸지 않는다.

| 계층 | 기존 필드와 의미 |
|---|---|
| Sample | `schema_version`, `sample_id`, `scene_family_id`, `split`, `mode`, `role`, `no_op_type`, `status` |
| Instruction | `text`, `language`, `selected_ids`, `locked_ids`, `group_by`, `sort_direction`, `axis_alignment`, `gap_m`, `region`, `keep_all`, `preference_text`, `expected_status`, `reason` |
| Scene | `family_id`, `seed`, `split`, `objects` |
| ObjectState | `object_id`, `asset`, `color`, `source_pose`, `target_pose`, `locked`, `move_label` |
| Asset | ID/Family, shape, a/b/h meter, footprint, mass, friction, provenance, license |
| 관계 라벨 | `rows`, `relations.horizontal_neighbors`, `relations.vertical_row_neighbors` |
| 렌더·감사 | Source/Target render directory, validation, source_validation, render_validation, file_hashes |
| 파생 물체 라벨 | instance_id, asset/family, color/shape, dimensions_m, Source/Target uv·pixel 중심, Envelope/Mask Box, locked/move |

마지막 행의 파생 라벨은 별도 정답 원본이 아니다. 원본 Pose·Asset에서 다시 계산할 수 있어야 한다. Object ID로 Join하고, 배열 순서만으로 Source와 Target을 대응시키지 않는다. Instance ID는 Render Metadata의 명시적 매핑을 따른다. 0은 배경, 1..N은 물체다.

`move_label`은 중심 이동 >3 mm 또는 π 주기 방향 변화 >3°일 때 true이다. `locked`와 다르다. 선택 물체도 이동이 불필요하면 move=false일 수 있다.

모델 입력은 Source RGB와 `instruction.text`뿐이다. selected/locked IDs, 구조화된 의미 필드, GT 크기·Pose·Mask·Target RGB, Teacher Row, Object ID는 감독·평가용이고 NN 입력에 넣지 않는다. `preference_text`는 현재 빈 예약 필드이며, Preference는 명시적 지시 문장에 포함한다.

### 4.2 추가할 Metadata

기존 Scene/Instruction Dataclass에 임의 필드를 끼워 넣지 않는다. 다음을 Dataset 수준 `provenance.json` 및 별도 `teacher_audit.jsonl`, `migration.jsonl`, `human_review.jsonl`에 추가한다.

- 활성 데이터: `spec_version="1.3"`, `teacher_policy="balanced_rows_v1.3"`, `instruction_contract="english_rows_v1"`, `language_policy="english_only_v1"`, `supported_languages=["en"]`. 기존 `row_edges_v1`은 Legacy 문장 계약으로 보존한다.
- `parent_dataset_manifest_sha256`, Teacher 코드 Revision, 설정 Hash.
- Sample별 기존/신규 Target Hash, 변경 종류, Template/행 크기, J_shape, J_move, 검사 수, 실패 사유.
- Counterfactual Pair ID, 바뀐 의미 항목, 교차 지시 검증 결과.
- 검토자, 검토 일시, Sample별 의미·정돈감·수정 자연스러움·가독성 판정.
- 추후 연속 수정 데이터에는 lineage_root와 parent_sample_id를 별도 Metadata로 기록한다.

정책 Metadata가 없는 기존 Dataset은 출처를 확인한 뒤 명시적으로 Legacy 정책으로 등록한다. Loader가 현재 코드 버전만 보고 Legacy Target을 v1.3으로 해석하면 안 된다. Checkpoint에는 사용한 Manifest Hash와 Teacher 정책을 저장한다.

## 5. M0–M8 구성과 변경 위치

| Module | Input → Output | Dependencies | v1.3 작업 |
|---|---|---|---|
| M0 Schema | Asset/Pose/Instruction → 검증된 구조 | 표준 라이브러리 | 기존 필드·단위 유지 |
| M1 Assets | Seed·치수 → Registry/Mesh/Footprint | M0 | 유지 |
| M2 Scenes | Registry·Seed·Split·N → Source Scene | M0/M1/Geometry | 기존 생성 보존; 연속 수정 Source는 후속 범위 |
| M3 Instructions | Scene·의미 선택 → 영어 문장/의미 라벨 | M0 | English-only 기본값; Legacy 문법은 명시적 회귀 재현용 |
| M4 Teacher | Scene+Instruction → status/Target/rows/relations | M0, 배치 기하, M5 | 후보·선택 점수·Partial 이동 기준 수정 |
| M5 Validator | Source/Target/Instruction → 독립 판정 | M0, Shapely | 기존 기하·의미 계약 유지; 변경 규칙과 회귀 검사 |
| M6 Rendering | Pose+Asset+Camera → RGB/Mask/Metadata | Blender bpy | 유지; 바뀐 Target만 렌더 |
| M7 Builder | Source/Instruction/Teacher 결과 → Annotation/Manifest | M0–M6 | 고정 입력 Migration, 정책 Metadata, 변경 Target만 갱신 |
| M8 Audit/QC | Dataset → 감사/분포/검토 자료 | M0/M5/M6/M7 | 정책 혼합·결정성·Counterfactual·품질 비교 추가 |

기존 `build_dataset`은 비어 있는 출력 디렉터리와 최대 100 Family를 요구한다. 이를 우회해 대량 생성하지 않는다. Migration은 새 경로를 대상으로 하고, Source를 다시 샘플링하는 `prepare_annotations(seed)`와 구분된 작업이어야 한다. Teacher 변경은 Seed 기반 채택 순서까지 바꿀 수 있으므로 동일 Seed 전체 재생성은 기존 Scene 보존 절차가 아니다.

## 6. Heuristic Teacher의 확정 규칙

### 6.1 공통 의미

선택 집합 S와 고정 집합 K는 모든 물체를 중복 없이 분할한다. Partial의 Row·정렬·영역 조건은 S에 적용하고, 전체 간격·충돌·경계는 S∪K에 적용한다. K의 Pose는 정확히 복사한다.

크기 정렬은 외접 직사각형 면적 a×b 기준이다. 색상 그룹이 있으면 각 색상 안에서 정렬한다. 없는 경우 선택 물체 전체에서 정렬한다. Teacher는 정확한 값으로 정렬하고, Evaluation의 5% 역전 허용은 수용 판정에만 쓴다. 색상 순서는 Teacher에서 red→green→blue로 고정하지만, 지시가 순서를 정하지 않았으면 Evaluation은 다른 연속 그룹 순서도 허용한다.

정렬 Key는 `(sign·a·b, a, b, color, source_y, source_x)`로 고정한다. Shape/Asset ID나 입력 배열 순서는 Tie-break에 쓰지 않는다. 완전 동률에 따른 관측 불가능한 대응은 생성 범위에서 제외한다.

### 6.2 No-op 우선

현재 Scene이 해당 지시와 최종 안전 조건을 이미 만족하면 모든 Target Pose를 정확히 Source Pose로 복사한다. `rows=[]`, `relations={}`, 모든 locked=true, move=false다. Teacher의 새로운 균형 점수를 이유로 이미 올바른 배치를 다시 움직이지 않는다.

명시적 Keep-all은 안전한 산개 장면도 보존할 수 있다. 반복 정돈 지시는 요청된 의미 조건까지 만족해야 한다. 초기 간격이 5 mm 이상 10 mm 미만이고 수치 허용오차 밖인 장면은 정상 No-op 정답이 아니다. Keep-all과 최종 안전 조건이 함께 만족될 수 없으면 infeasible이다.

**위치 기반 참조의 재적용:** Teacher에 같은 구조화된 Instruction(`selected_ids`/`locked_ids`와 요청 조건을 유지)을 결과 Scene에 재적용하는 검사는 동일 대상에 대한 멱등성 검사다. 반면 새 관측에서 “the leftmost object” 같은 자연어를 다시 해석하면 이동 후 다른 물체를 가리킬 수 있다. 이 경우 선택·고정 집합을 현재 관측 기준으로 다시 결정하며, 이전 ID를 숨은 대화 기억처럼 유지하지 않는다. 선택 집합이 바뀐 재해석은 새로운 의미 입력이므로 무조건 No-op을 요구하지 않는다. 새로 해석한 지시와 최종 안전 조건을 현재 Scene이 이미 만족할 때만 No-op이다. 위치 참조가 동률 등으로 모호해지면 기존 ambiguous 기준을 적용한다.

### 6.3 후보 확장

No-op이 아닌 경우 선택 물체 수 N_s에 대해 k=1..N_s를 사용한다. 색상별 순서를 유지한 채 각 그룹에서 다음 두 분할을 만든다.

1. 기존 순차 분할: k개씩 잘라 마지막 나머지를 둔다.
2. 균형 분할: 그룹 크기 n에 대해 r=ceil(n/k), q=floor(n/r), s=n mod r. 앞의 s개 행은 q+1개, 나머지는 q개를 배치한다.

예를 들어 n=5, k=4이면 기존 4+1과 균형 3+2를 모두 남긴다. 한 Template에서는 모든 색상 그룹에 같은 k와 분할 방식을 적용한다. 색상별 조합을 전부 탐색하지 않는다. 선택 순서의 Index Tuple로 중복 Template을 제거한다. 기본 후보는 최대 2N_s개로 제한된다.

Partial에는 관측된 기존 Row 구성을 유지하는 Template 하나를 추가한다. M5와 같은 Row 판정으로 선택 물체만 읽고, 색상 순수성·연속성·크기 순서가 새 지시와 맞을 때만 기존 행별 구성원을 유지한다. 새 방향·간격으로 치수를 다시 계산한다. 맞지 않으면 이 후보는 생략한다. 완전한 기존 Row 구성이 없는 Source에 임의로 과거 구조를 추론하지 않는다.

### 6.4 후보 좌표

각 물체의 w,h는 목표 방향에서 계산한다. 같은 Row의 중심 y는 동일하고, x는 다음을 만족한다.

\[
x_{i+1}=x_i+w_i/2+g+w_{i+1}/2.
\]

Row 높이는 max(h), Row 폭은 Σw+(n_row−1)g이다. 모든 Row의 x 중심은 배치 Block의 공통 중심이다. 위에서 아래로 Row를 배치한다. 같은 색상 그룹의 Row 띠 사이 간격은 g, 다른 색상 그룹 사이 간격은 2g다.

Row 띠는 해당 Row의 모든 Envelope를 감싸는 수평 띠이다. 이 간격은 서로 다른 Row에 있는 모든 개별 물체 쌍의 간격이 g라는 뜻이 아니다. 예를 들어 높이가 다른 물체가 섞이면 짧은 물체 사이에 더 큰 여유가 생긴다. 자연어와 QC에 이 의미를 유지한다.

### 6.5 전체 정돈 선택

요청 영역 중앙에 각 Template의 Block 중심을 놓고, 전체 기하·의미 검증을 통과한 후보를 비교한다. 사용 가능 영역 폭·높이를 R_w,R_h, Block 폭·높이를 B_w,B_h라 할 때:

\[
J_{shape}=\max(B_w/R_w,\ B_h/R_h).
\]

순서는 `(J_shape, B_w·B_h, 행 수, 고정 Template Key)`의 오름차순이다. 실수 비교는 J_shape·면적을 소수 12자리까지 반올림한 Key로 수행하고, 마지막에는 관측 기반 물체 순서로 정의된 Template Key를 사용한다. Raw 값은 감사 보고서에 보관한다.

이 기준은 영역 대비 지나친 가로/세로 확장을 줄이는 Engineering 선택이다. 미적 최적성을 보장하지 않는다. 100 Scene 비교에서 규칙적으로 품질이 나빠지면 대량 생성 전에 이 점수만 재검토한다.

### 6.6 부분 수정 선택

각 Template에 대해 Block 전체가 요청 영역에 들어오는 기존 전역 5 mm Translation 격자를 사용한다. 고정 물체는 움직이지 않는다. 후보별 비용은:

\[
J_{move}=\frac1{|S|}\sum_{i\in S}\|p_i^{target}-p_i^{source}\|_2^2.
\]

모든 Template/Translation 쌍을 `(J_move, J_shape, Block 면적, 행 수, Template Key, -center_y, center_x)` 순서로 정렬한 뒤 첫 유효 후보를 선택한다. J_move는 m²이고 비교 Key는 소수 12자리 반올림을 사용한다. 초기 구현은 이 전역 순서를 기준으로 예산을 소모한다. Template마다 따로 탐색하다 예산을 소진하는 구현은 결과를 바꿀 수 있으므로 사용하지 않는다.

영역 중심과 가까운 순서로 첫 Template의 첫 유효 배치를 반환하던 기준은 제거한다. 기존 Template와 5 mm 격자를 남기므로 기존 유효 Partial 후보도 후보 집합에 포함된다. 최소 이동은 이 유한 후보 집합 안에서의 의미이며 전역 연속 최적화를 뜻하지 않는다.

기하 검사 예산은 기존 20,000회를 초기값으로 유지한다. 비용 정렬은 기하 검사 수에 포함하지 않는다. 예산 내 유효 결과를 찾지 못하면 `generation_failure/search_budget`이다. 후보 전체가 실패해도 `candidate_family_exhausted`이지 infeasible의 증거가 아니다.

### 6.7 결과 일관성

같은 관측 Scene와 같은 의미의 Instruction에는 같은 ID별 Target을 반환한다. 언어, 문장 바꿔 쓰기, 조명, 배경, 배열 순서, Asset ID는 배치 정책의 난수로 사용하지 않는다. 물체 기하·현재 Pose·선택 집합·요청 조건이 변하면 정답이 바뀔 수 있다.

Teacher 결과에 선택·고정 집합과 요청 조건이 같은 구조화된 지시를 다시 적용하면 No-op이어야 한다. 자연어를 새 관측에 다시 grounding하는 경우에는 6.2절을 따른다. 위치 기반 참조의 대상 변경을 결정성 실패나 No-op 실패로 오인하지 않는다. 기존 ID를 유지한 재적용 검사는 새로운 이미지의 위치 참조를 재해석하는 기능까지 검증한 것이 아니다. 1 mm 치수 변화에 Row 구성이 바뀌는 것은 임계점에서 가능한 현상이다. 이런 전환은 태그·시각 검토 대상으로 기록하되 동일 입력에 무작위 정답을 추가하지 않는다.

## 7. 현재 Teacher와의 비교

기준은 `69cf565`의 [teacher.py](../src/knolling_dataset/teacher.py), [instructions.py](../src/knolling_dataset/instructions.py), [builder.py](../src/knolling_dataset/builder.py), [validator.py](../src/knolling_dataset/validator.py)다.

| 항목 | 현재 충족 여부 | v1.3 조치 |
|---|---|---|
| 목표 회전 후 a/b 교환 | 충족 | 유지; 일반식과 0/90° Fixture 대조 |
| 물체 경계 간격·Row max-height | 충족 | 유지 |
| 색상별 독립 Row·연속 그룹 | 충족 | 유지 |
| 외접 면적과 그룹 내 정렬 명시 | 충족 | 유지 |
| No-op 정확 복사·빈 관계 라벨 | 충족 | 유지 |
| Source–Target ID·고정 Pose 보존 | 충족 | 유지 |
| 후보 k=1..4와 단일 Chunk | 미충족 | k=1..N_s, 기존+균형 분할 |
| 행 수 우선 점수 | 미충족 | J_shape 우선 |
| Partial 영역 중심 우선 | 미충족 | J_move 우선, 기존 Row 후보 |
| 동일 Seed의 결정성 | 기본 구현 충족 | 배열/언어/재적용 불변성 검사 강화 |
| Row 간격 문장 | 수치는 일치; Row 띠 표현은 불명확 | 활성 영어 문장에 Row 띠 의미 명확화; 숫자·의미 필드 유지 |
| Counterfactual | 30 Family 존재, 주로 방향 전환; region도 full로 변경될 수 있음 | 변경 의미 항목 기록, 교차 조건 검증, 후속 확장에서 단일 요인 Pair 확보 |
| 실패와 infeasible 구분 | 충족 | 유지 |
| 독립 기하 Validator | 충족 | 유지; Teacher 점수와 결합하지 않음 |
| 배치 품질 전수 사람 검토 | 미완료 | 130 Sample 검토·기록 |

기존 Counterfactual이 두 의미를 동시에 바꿨다는 이유만으로 오류는 아니다. 다만 방향/간격/영역 각각의 언어 이해를 분리 진단하려면 후속 확장 때 단일 요인 Pair가 필요하다.

## 8. 기존 Pilot의 보존과 Migration

### 8.1 적용 절차

1. 기존 `pilot_v1_2`의 Manifest/Provenance/감사 결과를 기준 기록으로 고정한다. 원본 파일과 기존 검토 상태를 수정하지 않는다.
2. 130개 기존 Annotation에서 Source·Asset·Instruction 의미 필드·ID·Split을 읽는다. Source 샘플링을 다시 하지 않는다.
3. v1.3 Teacher를 같은 입력에 적용하고 Target을 ID별로 비교한다. No-op은 기존 Source의 의미 충족 여부로 판정하므로 Legacy 정돈 모양도 보존할 수 있다.
4. 변화가 없는 Sample, Target 변화 Sample, 문장만 명확화한 Sample, 생성 실패를 구분해 Migration 보고서에 기록한다.
5. 새 Dataset 경로에 기존 Source RGB/Mask/Metadata를 해시 검증 후 복사한다. 원본을 위험하게 공유하는 가변 Hard Link는 사용하지 않는다.
6. Target이 같으면 기존 Target Render를 재사용한다. 달라지면 해당 Target만 렌더한다. No-op은 Source Render를 재사용한다.
7. 변경 Sample의 Target Pose, 파생 uv/pixel/Mask Box, rows, relations, locked/move, validation, file_hashes, Annotation Hash, Manifest, Split Index, QC를 함께 갱신한다.
8. 130개 전수 감사와 사람 검토를 완료한 뒤 새 완료 기록을 만든다. 실패한 항목을 숨기고 v1.3 Dataset 완료로 표시하지 않는다.

`generation.json`의 100/101은 Legacy 원본 생성 이력으로 보존한다. 고정 입력 130개 Migration의 통과율은 별도 기록한다. v1.3 Target을 적용했다고 기존 채택률을 새 Teacher의 생성 성능으로 재표기하지 않는다.

### 8.2 무엇을 다시 만들어야 하는가

| 변화 | 필요한 작업 | 불필요한 작업 |
|---|---|---|
| Teacher 결과 동일 | 새 정책에서 재검증·정책 기록 | Source/Target 재렌더 |
| Target Pose 변경 | Target Annotation·Render·파생 라벨·감사 갱신 | Source/Asset/Split 재생성 |
| 문장 명확화만 | Text/Annotation Hash/Manifest/QC 갱신·의미 동일 확인 | 이미지 재렌더 |
| 새로운 의미의 지시 추가 | 같은 Source의 새 Sample·새 Target 생성 | 기존 Sample 덮어쓰기 |
| 새로운 후속 관측 Scene | 새 Source ID/Render·lineage 기록 | 기존 Family의 Source 교체 |

현재 M8은 하나의 `scene_family_id`에 하나의 Source 물리 장면만 허용한다. 후속 관측은 새로운 Scene Family ID를 부여하고 원래 lineage_root 및 Split을 상속한다. 원래 Family의 Source를 교체하면 `family_source_drift`가 발생한다. Episode Split 검사는 lineage_root까지 확장한다. 이 확장은 초기 Teacher Migration의 필수 작업이 아니라 Interactive 단계의 작업이다.

### 8.3 Legacy 호환성 검사

`tests/test_pilot.py`의 현재 재생성 검사는 현재 `prepare_annotations`를 호출해 v1.2와 정확히 비교한다. 기본 Teacher를 바꾸면 이 검사는 의도적으로 달라진 Target까지 회귀 오류로 오인할 수 있다.

후속 구현에서는 기존 규칙을 `legacy_v1_2` 정책으로 명시적으로 보존하고 기존 재현 검사는 이 정책을 호출하도록 한다. v1.3은 별도 정책 재현 검사를 둔다. Legacy Dataset의 독립 기하·해시 감사는 계속 통과해야 한다. 기존 테스트를 삭제하거나 Expected 값을 새 Target으로 덮어써 원본 이력을 지우지 않는다.

학습 Loader는 단일 활성 Teacher 정책을 요구한다. v1.3 재평가에서 Target이 같았던 Legacy Sample은 v1.3 Manifest에서 재사용할 수 있지만, 서로 다른 정책이 낳은 경쟁 Target을 한 학습 집합에 혼합하지 않는다. Legacy 비교용 학습은 별도 Run으로만 관리한다.

### 8.4 영어 전용 문장 Migration

`pilot_v1_3`는 수정하지 않고 `pilot_v1_3_en`을 새 디렉터리에 만든다. 활성 Annotation에서는 `instruction.text`와 `instruction.language`만 변경한다. 나머지 의미 필드, Source/Target Pose, 물체 ID·자산·크기, 관계, 이동/잠금 라벨, Scene Family 및 Split은 정확히 보존한다. Source/Target RGB·Mask·Render Metadata는 해시가 같은 독립 파일로 복사하고 Blender를 호출하지 않는다. QC Overlay는 검토용 파생 자료이며 모델 입력 이미지를 변경하지 않는다.

Annotation Hash와 전체/분할 Manifest 및 QC를 새로 작성한다. `language_migration.jsonl`에는 Sample별 이전 언어, 텍스트 Hash, 의미 Hash 및 영어 길이를 기록하고, `language_migration_summary.json`은 활성 언어 수·의미 불일치·0건의 재렌더를 보고한다. 원본 생성·기하 Migration 통계는 역사 기록으로 보존하고 현재 언어 분포와 혼동하지 않는다. `language_parent_manifest.jsonl`, `language_parent_annotations/` 및 기존 `parent_annotations/`는 출처 검증용이며 학습에서 제외한다. Loader는 활성 `manifest.jsonl`과 그 Split Index만 사용하고 디렉터리 전체 Annotation을 glob하지 않는다.

언어 변경 후 사람 검토 상태는 pending으로 두고 `training_approved.jsonl`은 비운다. 코드와 CLI의 새 생성 기본값은 balanced Teacher + English-only다. 기본 감사도 영어 계약을 요구한다. Legacy 회귀 감사는 명시적 `--legacy --read-only`로 분리하며 활성 영어 데이터의 비영어 검사를 끄는 용도로 사용할 수 없다.

## 9. Dataset 생성과 품질 검증

### 9.1 Pipeline

Asset/Geometry Family Split 고정 → Source 생성/읽기 → 의미 조건 선택 → 영어 문장 → Teacher → 독립 Validator → Blender 렌더 → RGB/Mask/투영 검사 → Annotation/Manifest → 전체 감사 → 사람 검토 순서다. 기존 자료의 언어만 변환할 때는 8.4절에 따라 Teacher 배치 생성과 렌더 단계를 생략한다.

Blender 직접 bpy Backend를 유지한다. 기존 환경은 Blender 5.1.0, Host Python 3.12.16이다. RGB는 Cycles CPU 32 Samples, Instance Mask는 같은 Mesh/Camera의 Workbench flat object color, anti-aliasing 비활성화 후 uint16 ID 변환이다. PNG 시각/렌더 시간 Metadata 정규화를 유지한다. Bitwise 재현성 주장은 같은 Blender Build·장치·플랫폼에 한정한다. BlenderProc를 다시 도입하지 않는다.

카메라 보정·해상도·출력 방향은 기존 Render Job 계약을 그대로 사용한다. 초기 Neural 학습에서는 임의 Crop/회전/좌우 반전 Augmentation을 사용하지 않는다. 그런 변환은 좌표와 “left/right” 지시까지 함께 바꿔야 한다. 색상 이름이 바뀌는 Hue Augmentation도 사용하지 않는다.

### 9.2 보존할 자동 기준

- 활성 네 Split 모두 `language="en"`이고, Source와 구조화된 필드에서 만든 승인된 영어 문법과 텍스트가 일치해야 한다. 언어 필드만 en으로 바꾼 한국어, ASCII로 쓴 다른 언어, 잘못된 숫자·대상·방향도 거부한다. 이는 제한 문법의 정합성 검사이며 범용 언어 감지기를 주장하지 않는다.
- 원본 대비 text/language 외 필드 및 Source/Target 파일 Hash가 같고, 각 Split Index가 새 Manifest의 해당 부분집합과 정확히 일치해야 한다.
- 물체 ID·개수·Asset·색상 보존, 고정 Pose 정확 복사, 단위/좌표/Relation 재계산.
- Target 간격 ≥10 mm, 경계 ≥10 mm, Envelope 충돌 없음, 지시 영역 포함.
- Mask–실제 Footprint IoU ≥0.98, 중심 투영 오차 ≤1 pixel.
- 내부 RGB 색상 일치 ≥99%, clipped white pixel ≤1%.
- Source/Target Instance ID 매핑 일치; 학습 RGB에 QC ID·화살표 포함 금지.
- Scene/Asset/정규화 형상/동일 물리 장면의 Split 중복 0.
- Source/Instruction 배열 순서 변경, 같은 의미의 문장, 동일 선택·고정 집합과 요청 조건을 유지한 구조화된 지시 재적용에 대한 Target 일관성. 새 관측에서 위치 참조를 재해석하는 검사는 별도로 구분한다(6.2절).
- 잘못된 ID·회전 치수·고정 Pose·단위·5–10 mm No-op을 넣은 Fixture를 Validator가 거부.

### 9.3 다양성과 언어 검증

활성 언어 분포는 영어 100%다. 영어 지시 유형·관측 가능한 참조·문장 의미 변화와 Counterfactual을 평가한다. Legacy 한국어 회귀 검사를 활성 다국어 평가나 학습 표본으로 집계하지 않는다.

N=2..8, 세 형태·색상, 종횡비·크기 조합, 수평/수직, 15/20/30 mm, 전체/좌/우 영역을 조건별로 집계한다. 색상과 크기·형태가 고정적으로 연관되지 않게 한다. 단색/두 색/세 색과 5+2+1 같은 불균형 구성도 포함한다. 특정 Row 빈도를 억지로 균등하게 만들지 않고 N·색상 구성·영역 안에서 분포를 비교한다.

Counterfactual Pair에서는 의미 필드 차이뿐 아니라 각 Target을 반대편 Instruction으로 평가한다. 진단용 Pair는 양쪽 Target이 각각 반대 지시의 적어도 한 조건을 위반해야 한다. 기존 Pair가 이 기준을 통과하지 못해도 개별 Sample의 정답은 유지하고, 유효한 언어 진단 Pair 수에서 제외한다. 필요한 부족 Pair만 추가한다.

현재 100 Scene은 70/10/10/10 Family Split, 89/13/13/15 Sample Split이다. 이는 작은 Geometry-OOD 정합성 Pilot이다. 낮은 표본 수로 95% 이상의 Neural 성공률을 입증하지 않는다. Validation은 Checkpoint 선택, Development는 실패 진단, Test는 고정 정책의 최종 평가에 사용한다. 기존 100 Scene Test는 데이터 QA에 이미 노출됐으므로 향후 최종 성능 주장은 새로 잠근 Test 집합에서 평가한다.

### 9.4 데이터 확장 순서

초기 100 Scene은 형식·품질 검증과 Train Split 안의 작은 Overfit에 사용한다. 학습 일반화 Pilot의 계획량은 Train 2,000 / Validation 400 / Development 1,000 / 잠금 Test 1,000 Sample이며 N=2–4부터 시작한다. 같은 Family의 모든 영어 표현·이미지·명령 파생형은 같은 Split에 둔다. 다국어 파생형은 현재 생성하지 않는다. Source가 바뀐 Episode도 lineage 단위 Split을 유지한다.

초기 Pilot에서도 Full/Partial/No-op 기본 의미와 No-op 학습은 유지하되, 하나의 기본 정돈 지시부터 시작하는 Curriculum을 허용한다. Geometry-OOD Split은 유지하고 다른 Split의 Asset을 성능 향상을 위해 Train으로 옮기지 않는다.

후속 목표량 50,000 Positive(Full 25,000, Partial 15,000, No-op 10,000) 및 Status용 5,000 Negative는 확장 예산이며 지금 생성할 의무가 아니다. 초기 Neural Pilot과 저장·처리량·Chunk/Resume 검증을 통과한 후 확장한다. Negative 계획은 ambiguous 2,000 / infeasible 1,500 / unsupported 1,500이며, 탐색 실패를 infeasible로 라벨링하지 않는다.

Partial의 현재 30개 산개 Source는 유지한다. 정돈된 상태에서의 수정과 3-turn 평가는 후속 Interactive 단계에서 별도 Episode를 보충한다. 현재 100 Scene이 그 기능까지 검증했다고 보고하지 않는다.

## 10. End-to-End Neural Architecture

### 10.1 연결 구조

Source RGB → Image Backbone/FPN → Object Decoder → 같은 Query의 Source 예측/물체 Token → 3단계 Arrangement Decoder → 같은 Query의 Target Pose.

자연어 → Token Encoder/Projection → Arrangement Decoder의 Text Cross-attention.

Status Head는 예약된 별도 출력이며 초기에는 학습·판정에 사용하지 않는다. 외부 Detector, 의미 Parser, Teacher, Packing Optimizer, GT 물체 수를 추론 경로에 넣지 않는다.

### 10.2 구현 기본값

| 구성 | v1.3 계약 |
|---|---|
| Image input | B×3×384×576, RGB/255와 Backbone 사전학습 mean/std 적용 |
| Backbone | ImageNet 사전학습 ResNet-18, 초기 Frozen BatchNorm 통계 |
| Image features | stride 8/16/32 FPN, 각 d=256; 2D 위치·Level Embedding |
| Object queries | Q=12, d=256 |
| Object Decoder | 4 Transformer Block; Query self-attention, Image cross-attention, FFN |
| 공통 Transformer | 8 heads, FFN=1024, GELU, dropout=0.1, Pre-LayerNorm |
| Text Encoder | paraphrase-multilingual-MiniLM-L12-v2의 Token feature 384차원, 선형 384→256 |
| Arrangement Decoder | 3 Block; Query self-attention, Text cross-attention, FFN; Object Token과 단계별 Geometry Embedding 사용 |
| Source heads | Objectness, center uv, 고유 a/b, π 주기 방향, color 3-class, shape 3-class |
| Target heads | 단계별 center uv·방향 residual |
| 선택 출력 | Objectness ≥0.5; NMS·GT top-k 사용 안 함 |

Backbone/FPN/Object Decoder/Arrangement Decoder는 통합 Graph다. 초기 Text Encoder 고정은 사전학습 Neural 표현을 사용하는 학습 정책이며 외부 Rule-based 언어 처리기가 아니다.

English-only 정책을 적용해도 위 Neural Architecture와 현재 Text Encoder 선택은 변경하지 않는다. Encoder 이름에 multilingual이 포함되어 있다는 사실은 초기 시스템의 다국어 지원을 뜻하지 않는다. 영어 전용 사전학습 Encoder의 이점은 추후 Neural Pilot에서 별도로 비교할 수 있지만 이번 Dataset 변경에 포함하지 않는다.

영어 지시를 간결하게 해도 **실제 Token 길이를 측정하고 silent truncation을 금지**한다. 최대 길이는 special token 포함 512, Batch 내 동적 Padding을 기본으로 유지한다. 모델 Config의 position limit 512를 넘는 입력은 오류로 보고한다. 단어 수를 Token 수로 간주하거나 의미를 몰래 생략하지 않는다. 이는 핵심 Network 구조 변경이 아니라 입력 보존 설정이다. [공식 모델 Config](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/raw/main/config.json)

### 10.3 Head 표현과 Refinement

Source center는 sigmoid로 uv를 출력한다. a/b는 양수이고 a≥b가 되도록 `b=softplus(z_b)+ε`, `a=b+softplus(z_delta)`로 파라미터화한다. ε=1e-4 m, 초기 Bias는 대략 a=0.060 m, b=0.035 m를 출력하도록 설정한다. GT 범위 밖의 예측도 Loss로 교정하며 추론에서 GT 크기로 대체하지 않는다.

Source 방향 Head는 두 실수 (c,s)를 출력하고 θ=0.5·atan2(s,c)로 읽는다. 초기 Bias=(1,0), 길이 0 부근에는 epsilon 안정화를 적용한다. Loss는 π 주기 각도 오차다. 별도 방향 정규화 Loss는 추가하지 않는다.

초기 Target Pose는 예측 Source Pose다. 각 단계 k의 Geometry Embedding은 예측 Source uv/a/b/방향/Objectness와 이전 Target uv/방향을 사용한다. 방향은 cos(2θ),sin(2θ), 크기는 a/0.6,b/0.4로 표현한다. GT 값이나 GT selected/locked Mask로 대체하지 않는다. 이 Embedding은 MLP로 256차원에 투영해 해당 Query Token에 더한다.

\[
\hat u^k=\sigma(\operatorname{logit}(\operatorname{clamp}(\hat u^{k-1},10^{-4},1-10^{-4}))+\alpha_k\Delta u^k),
\quad \alpha=(1,0.5,0.25).
\]

v도 동일하다. 방향은 θ^k=wrapπ(θ^(k−1)+(π/2)tanh(Δθ^k))로 갱신한다. Residual Head 마지막 Linear는 0으로 초기화한다. 초기 출력은 Source를 복사하지만, 모든 단계는 Target 감독으로 이동을 학습한다. 이 좌표 Step 계수와 Loss의 단계 Weight는 다른 값이다.

GT Matching은 Gradient 경로 밖의 할당 연산이다. 반면 Source 예측→Geometry Embedding→Target 및 단계 간 Refinement는 detach하지 않는다. 초기에는 모든 12 Query가 Attention에 참여하고, 예측 Objectness로 학습 중 GT 물체를 Hard-mask하지 않는다. 출력 정리는 최종 Objectness Threshold로만 한다.

별도 Keep/Move 판정이 Target을 강제로 Source로 바꾸는 구조는 사용하지 않는다. move_label은 진단용으로 유지한다. 물체 이동 필요 여부는 유효한 Source/Target 예측의 차이로 판정할 수 있다.

### 10.4 Neural Module 인터페이스

| Module | Input | Output | 의존성과 검증 |
|---|---|---|---|
| N0 Dataset Adapter | 고정 Manifest·Annotation·Source RGB | image tensor, token_ids, text_mask; 별도 supervision bundle | M0/M7/M8, Tokenizer; GT가 입력에 섞이지 않는지 확인 |
| N1 Visual/Object Model | image tensor | B×12×256 Object Token, Source Head tensors | Backbone/FPN/Object Decoder; Source Geometry 단위 확인 |
| N2 Arrangement Model | Object Token·예측 Source·Text Token | 3개 단계의 B×12×(uv,θ) | 같은 Query Index 유지, GT Size/Lock 미사용 |
| N3 Criterion | Source/Target 예측·GT supervision | 단일 Source Matching, scalar total, 항별 통계 | 11절, 유효 Sample Mask·Gradient 검사 |
| N4 Trainer | N0/N1/N2/N3·Run config | Checkpoint·학습/Validation 기록 | 12절, 데이터/정책 Hash 고정 |
| N5 Evaluator | Raw Prediction·평가 GT | Count·오차·Geometry·SSR·조건별 실패 | 14절; Dataset Validator의 exact 조건과 구분 |
| N6 Execution Adapter | 예측 Source/Target·Calibration | 승인/거부, 실행 결과, 재관측 RGB | 15절; Target 좌표 수정 금지 |

Inference Result의 물체 Key는 그 이미지의 `query_index`다. 각 항목은 Objectness, Source center/size/angle, Color/Shape 확률, 최종 Target center/angle을 포함한다. 이것을 GT object_id라고 부르지 않는다. 실제 집기는 예측 Source 위치를 이용하고, GT ID와의 대응은 평가에서만 만든다. 새 관측 이미지에서 Query 번호가 유지된다고 가정하지 않는다.

## 11. 초기 Loss와 Identity Matching

### 11.1 정규화

Smooth L1을 ρ(z)=z²/2 (|z|<1), |z|−1/2 (그 외)로 정의한다. meter로 복원한 오차를 s_p=.010 m, s_s=.005 m, s_θ=π/18 rad로 나눈다. 이는 Loss 척도이며 성공 허용오차가 아니다.

\[
\ell_p(\hat p,p)=\tfrac12\sum_{d=x,y}\rho((\hat p_d-p_d)/s_p),\qquad
\ell_s(\hat s,s)=\tfrac12\sum_{d=a,b}\rho((\hat s_d-s_d)/s_s).
\]

\[
\Delta_\pi(\hat\theta,\theta)=\tfrac12\operatorname{atan2}(\sin(2(\hat\theta-\theta)),\cos(2(\hat\theta-\theta))),\qquad
\ell_\theta=\rho(\Delta_\pi/s_\theta).
\]

179°와 1°의 오차는 2°다. 정확히 90° 차이는 주기적 거리의 비매끄러운 경계이므로 유한값·기울기 Fixture를 포함한다. [SmoothL1 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)

### 11.2 Source Matching

GT i와 마지막 Object Decoder Query j의 Cost:

\[
C_{ji}=-q_j+\frac{\|\hat p_j^s-p_i^s\|_1}{2s_p}
+\frac{\|\hat s_j-s_i\|_1}{2s_s}
+\frac{|\Delta_\pi(\hat\theta_j^s,\theta_i^s)|}{s_\theta}.
\]

Hungarian으로 GT N개와 서로 다른 Query N개를 대응한다. 한 Forward Pass의 이 π를 Detection 및 세 Target 단계 전체에 재사용한다. 다음 Forward Pass에서는 다시 계산한다. Target 좌표, Teacher Row, 목표 순서로 다시 Matching하지 않는다. Query 번호가 다른 Scene 사이에서 같은 물체를 뜻할 필요는 없다.

Color/Shape는 초기 Matching Cost에서 제외하고 할당 후 감독한다. 별도 Identity Contrastive Loss, GIoU, Query Denoising은 초기 필수 항목이 아니다. 일대일 Set Matching의 근거는 [DETR](https://arxiv.org/abs/2005.12872)이며, 위 물리 단위 Cost는 본 프로젝트의 고정 초기 설정이다.

### 11.3 Detection

Matched Query 집합 M, Objectness q, negative 가중치 η=.1에 대해:

\[
L_{obj}=\frac{-\sum_{j\in M}\log q_j-\eta\sum_{j\notin M}\log(1-q_j)}{N+\eta(Q-N)}.
\]

실제 구현은 Logit 기반 BCE를 사용한다.

\[
L_D=L_{obj}+\frac1N\sum_i[\ell_p(\hat p_{\pi(i)}^s,p_i^s)+\ell_s(\hat s_{\pi(i)},s_i)
+\ell_\theta(\hat\theta_{\pi(i)}^s,\theta_i^s)+.1CE_{color,i}+.1CE_{shape,i}].
\]

초기 Detection 감독은 마지막 Object Decoder 출력에 적용한다. 중간 Object Decoder 감독은 S0에서 마지막 출력의 수렴 문제를 확인한 경우에만 동일한 L_D를 같은 최종 Matching으로 평균하는 보완안으로 추가한다. 독립적인 층별 Weight 탐색은 하지 않는다.

### 11.4 Target과 전체 초기 목적함수

\[
L_T^{(k)}=\frac1N\sum_i[\ell_p(\hat p_{\pi(i)}^{t,k},p_i^t)+\ell_\theta(\hat\theta_{\pi(i)}^{t,k},\theta_i^t)],
\]

\[
L_T=.25L_T^{(1)}+.25L_T^{(2)}+.50L_T^{(3)},\qquad \boxed{L_{initial}=L_D+L_T}.
\]

모든 GT 물체를 Target Loss에 포함한다. Objectness가 낮다는 이유로 Matched Target 감독을 없애지 않는다. Partial/No-op의 고정 Target도 그대로 포함한다. 단계별 감독은 같은 목적의 Auxiliary Supervision이며 별도 세 목적함수가 아니다. Decoder 중간 감독의 선례는 [DETR 공식 구현](https://github.com/facebookresearch/detr/blob/main/models/detr.py)이다.

장면별 평균 후 Batch 평균한다. 서로 다른 N·Query 수·관계 수 때문에 Loss 총량이 자동으로 바뀌지 않게 한다. 기본 Batch Sampler는 Family를 균등 선택한 뒤 그 Family의 사용 가능한 Instruction Sample을 균등 선택한다. Counterfactual이 많은 Family가 과도하게 가중되지 않게 한다. 한 Epoch은 Train Family의 무작위 순열을 한 번 순회하는 것으로 정의하고, Family 안의 Instruction은 섞인 순환 목록으로 선택해 여러 Epoch에 걸쳐 모두 사용한다. 마지막 작은 Batch도 사용한다.

| 항목 | 초기 Weight |
|---|---:|
| Detection / Target | 1 / 1 |
| Detection 위치 / 크기 / 방향 | 각각 1 |
| Color / Shape CE | 각각 0.1 |
| unmatched Objectness | 0.1 |
| Target 단계 1/2/3 | 0.25 / 0.25 / 0.50 |
| Collision / Boundary / Relation / Keep / Status | 모두 0 |

## 12. 학습 단계와 기본 설정

### 12.1 Optimizer와 실행 기록

AdamW, Backbone LR=1e-5, Decoder/Head/Projection LR=1e-4, weight_decay=1e-4, Effective Batch=16, Global Gradient Norm Clip=1.0을 사용한다. Text Encoder는 초기 고정하고 Projection은 학습한다. 고정 Encoder는 eval 모드로 Dropout을 끈다. Bias/LayerNorm에는 Weight Decay를 적용하지 않는다. S0는 FP32, Augmentation 없음, 학습 모듈 Dropout=0으로 순수 과적합을 확인한다. S1부터 표준 Dropout=0.1을 적용하고 AMP는 그 후 수치 검증을 거쳐 선택한다.

단계 시작 1 Epoch LR Warmup 이후 해당 단계 예산 끝까지 Cosine Decay로 시작 LR의 0.1배까지 낮춘다. S0는 고정 LR이다. 매 Epoch Validation, 5 Epoch마다 Development 진단을 기록한다. Checkpoint에는 모델·Optimizer·Scheduler·RNG·Token 설정·데이터 Manifest/Teacher Policy·Loss Scale·Seed를 저장한다.

Checkpoint 선택은 전체 Task Success, 동률 시 Raw Geometry, 다시 동률이면 Target 중심 오차 순서다. S1에서는 Count 정확도 다음 Source 중심 오차를 쓴다. Raw 예측을 별도 Planner로 고친 결과로 선택하지 않는다. Test를 Weight 조정에 사용하지 않는다.

### 12.2 순서와 Gate

| 단계 | 목적 / 데이터 | Loss / 예산 | 통과 기준 |
|---|---|---|---|
| S0 | 검토된 Train의 32 Sample 과적합; Full·Partial·No-op 포함 | L_D+L_T, 최대 2,000 Update | Count 100%; Source/Target 중심 P95≤1 mm; 크기 P95≤1 mm; 방향 P95≤2° |
| S1 | N=2–4 Detection 준비; 계획 Train 2,000 | L_D, 최대 20 Epoch | Validation Count≥95%; Source 중심 P95≤5 mm; 크기 P95≤3 mm; 방향 P95≤5° |
| S2 | 단순 지시부터 End-to-End 공동 학습 | 5 Epoch λ_T:0→1, 이후 L_D+L_T; 첫 예산 총 100 Epoch | Raw Geometry≥90%, SSR≥80%; 첫 Seed 통과 후 총 3 Seeds 확인 |
| S3 | N=2–8·지시 범위 확장 | 확인된 최소 Loss; 기존 Checkpoint 계속 학습 | Count≥99%, Geometry≥97%, SSR≥90%, 지시별 SSR≥85% |
| S4 | Partial·No-op·후속 수정 | 기본 Loss; 필요 시 Keep | Partial 성공≥90%, No-op≥98%, 고정 물체 위반 Scene≤2% |
| S5 | 잠근 Test 최종 평가 | 학습 완료 후 평가 | Count≥99%, Geometry≥99%, SSR≥95%, 지시별≥90%, Geometry-OOD≥85%, 3-turn 전체 성공≥85% |

S0 수치는 Memorization 진단이며 일반화 성능이 아니다. S0에서도 안전·의미 오류 수를 별도 기록하고 잔여 실패를 확인한다. S0 실패를 보조 기하 Loss로 가리지 않는다. 단위·Matching·Target ID·Mask·Gradient 연결을 먼저 수정한다. S0 완료 후 정식 학습은 같은 사전학습 초기화에서 새 Run으로 시작한다.

S1에서 예측 Source가 준비되면 S2로 간다. 예산을 다 써도 Gate를 통과하지 못하면 Epoch만 늘리지 않고 실패를 분류한다. N 확대와 새로운 지시를 한꺼번에 적용하지 않고 한 축씩 확장한다. 기존 데이터의 선택·Sampler 변경은 가능하지만 Teacher/GT를 Model에 맞게 고치지 않는다.

현재 Split은 이미 Geometry-OOD다. 같은 Test 집합에서 SSR≥95%와 OOD≥85%를 따로 적용해 85%로 통과시켜서는 안 된다. 별도의 더 어려운 범위 내 Stress 집합을 사전에 정의했을 때만 85% 기준을 별도로 사용한다. 3-turn 기준은 Episode 데이터와 실행 재관측이 준비된 후에만 평가하며, 단일 장면 Neural Pilot의 선행 Gate가 아니다.

S2 이후에는 Source→Target Gradient를 유지한다. 공동 학습으로 Detection이 악화되면 첫 제한 조정은 λ_T=0.5다. 대규모 Weight Grid Search나 자동 Multi-task Weighting을 도입하지 않는다.

### 12.3 실패 진단과 최소 대안

| 관측된 실패 | 우선 확인 | 허용하는 다음 조치 |
|---|---|---|
| S0에서 Loss가 내려가지 않음 | meter/uv·각도 주기·Source Matching·Target ID·Gradient·출력 범위 | 오류 수정 후 같은 32 Sample 재확인; Loss 종류 추가 금지 |
| Source 누락/중복 | Objectness Target·unmatched 평균·12 Query·입력 Mask | 기존 Object Decoder 중간 출력에 같은 Detection 감독 평균 적용 |
| Source 정확, Target이 멈춤 | Residual 초기화 이후 Gradient·단계 간 detach·텍스트 Token 길이 | 연결/Mask 수정; 임시 GT 입력은 진단용 별도 Oracle 지표로만 사용하고 학습 경로에 넣지 않음 |
| 명령을 바꿔도 같은 배치 | 텍스트 잘림·Cross-attention Mask·Counterfactual 정답 차이 | 입력 오류 수정; 지속될 때 Text Encoder 마지막 2 Block을 LR=1e-5로 해제하는 한 가지 대안 |
| 공동 학습에서 Source 악화 | Target Gradient와 LR, Source Metrics | λ_T=0.5로 제한 비교 |
| Source 정상, 간격/경계만 실패 | 오차 크기·조건별 여유·최종 Footprint | 13절 해당 항 하나만 추가 |
| Train 성공, Validation 실패 | Split/형상 범위·과적합·이미지/문장 분포 | Coverage와 허용 Augmentation 점검; Test로 튜닝 금지 |

이 조치 후에도 3 Seeds에서 S2 Gate에 도달하지 못하면 추가 Loss를 쌓는 대신 실패 사례와 학습 곡선을 정리해 Architecture 재검토로 돌아간다. 다른 모델이나 End-to-End 조건 완화는 v1.3에 자동 포함하지 않는다.

## 13. 보조 Loss의 조건부 도입

### 13.1 공통 원칙

3회 연속 Validation에서 같은 실패가 반복될 때 검토한다. Detection이 맞는 장면에서 원인을 분리하되, 전체 SSR에서는 누락·오검출을 실패로 포함한다. 조건부 진단 표본이 100 Scene보다 적으면 작은 100-Scene 데이터의 몇 건만으로 비율 Threshold를 확정하지 말고 사례 수와 원인을 먼저 검토한다.

한 번에 한 항만 추가한다. 같은 Checkpoint에서 보조항 없이 동일 Epoch를 계속 학습한 대조 Run과 비교한다. 추가 Weight는 5 Epoch 동안 0→1, 이후 1로 고정한다. 정규화·Reduction을 아래와 같이 정의했을 때의 값이다. 다른 원시 Loss에 Weight만 복사하지 않는다. 영향이 과도하면 제한된 대안 0.3만 비교하고, 개선 없으면 제거한다.

| 항 | 추가 검토 조건 | 먼저 배제할 원인 |
|---|---|---|
| Collision | Source 인식 정상 장면에서 10 mm 간격 위반 >2% 지속 | 누락·크기 오류·Identity 오류 |
| Boundary | 경계/요청 영역 위반 >1% 지속 | 좌표 변환·중심만 검사하는 오류 |
| Relation | Target 위치는 개선되었지만 특정 간격/Row 관계 실패 >10% 지속 | 텍스트 잘림·명령 무시·조건 Label 오류 |
| Keep | Partial 고정 위반 Scene>2% 또는 No-op<98% | GT 복사·Lock·각도 주기 오류 |
| Status | Negative 지시의 거절 기능을 학습하는 단계 | Negative 데이터 부재·잘못된 infeasible Label |

Status는 기하 실패 보정항이 아니다. 유효 요청만 평가하는 초기 Pilot에서는 학습하지 않은 Status 출력으로 실행 승인/거절을 하지 않는다. 최종 거절 기능을 주장하려면 별도 학습·평가가 필요하다.

### 13.2 기하 Loss

최종 Target 단계에 먼저 적용한다. GT a,b와 예측 θ,p로 Rectangle을 만들고 GT 물체 전부를 포함한다. 예측 크기로 Collision을 계산해 물체를 축소하는 우회를 허용하지 않는다. 이 GT 사용은 감독에만 해당하며 추론 경로는 계속 예측 크기를 사용한다.

Collision은 L_C=mean_pairs ρ(max(0, .010−d_signed)/s_p)다. d_signed는 비겹침에서는 Rectangle Polygon 간 실제 Euclidean 최단거리, 겹침에서는 SAT로 구한 최소 분리 이동 깊이의 음수다. 외부 최단거리는 모서리–선분 거리의 최솟값으로 미분 가능한 Tensor 연산을 구성하고, 겹침 깊이는 Rectangle의 네 단위 축에서 계산한다. 축 평행/접촉/포함/대각선 분리 Fixture를 포함한다. SAT 투영 간격만을 비겹침 최단거리로 사용하지 않는다.

겹침 시 축 n에서의 분리 깊이는 r_i(n)+r_j(n)−|(p_i−p_j)·n|이며 네 축 중 최솟값을 사용한다. r는 해당 축으로 투영한 Rectangle의 반경이다. 단순 구간 교집합 길이는 한 Rectangle이 다른 Rectangle 안에 포함된 경우 이 값과 다르므로 사용하지 않는다.

Boundary는 Rectangle의 xmin/ymin/xmax/ymax에 대해 테이블 inset 10 mm 밖으로 나간 네 방향 부족량을 계산하고 ρ(부족량/s_p)를 방향·물체 평균한다. 선택 물체에는 요청 영역의 네 경계 부족량도 같은 방식으로 계산해 더한다. 두 부분은 각각 물체 평균 후 합산한다. 고정 물체에 요청 영역 조건을 강요하지 않는다.

Loss는 물체 간 실제 실행 안전을 증명하지 않는다. 활성화 전 모든 해당 GT에서 L_C와 L_B가 수치 오차 이내 0인지 확인한다. 완전 겹침처럼 기하 Gradient가 대칭인 경우에도 Target 감독은 유지한다.

### 13.3 Relation과 Keep

Relation은 기존 `relations`의 수평 이웃 Edge Gap, 인접 Row 띠 Gap에 대해 mean ρ((예측−GT)/s_p)를 적용한다. Row 중심 정렬 실패를 보완해야 할 때만 GT Row 안의 y 차이를 같은 척도로 추가한다. 활성 관계 종류는 Run 설정에 고정하고 각 종류 평균을 다시 평균한다. No-op과 관계가 없는 Sample은 0이며, Batch 정규화에서 해당 항의 유효 Sample만 평균한다. 이미 허용 범위의 No-op을 정확한 간격으로 재정렬하지 않는다.

Keep은 고정 집합 K에 대해 최종 Target과 GT Source의 ℓ_p+ℓ_θ를 평균한다. K가 없으면 0이다. 이는 Target 감독의 고정 물체 비중을 높이는 항이며 새로운 정답 정보가 아니다. 예측 Source와 비교해 두 Pose가 함께 틀린 것을 숨기지 않는다.

### 13.4 Status

기존 4-class Head(ok/ambiguous/infeasible/unsupported)를 사용한다. Token을 평균 Pooling한 Object/Text 표현을 연결해 MLP로 출력하며 초기에는 Head만 예약하고 Weight=0이다. 기존 렌더 Pilot은 전부 Positive이므로 Status 학습 완료 근거가 될 수 없다.

승인된 Negative 집합이 준비되면 CE를 사용하고 Class Weight는 Train 빈도의 역수로 고정한다. 표본 가중 평균의 분모는 Weight 합이다. 유효 Target이 없는 Negative는 Target/Collision/Boundary/Relation/Keep에서 제외한다. Source 정답이 있으면 Detection은 학습한다. Positive와 Negative의 Loss 유효 Mask를 명시한다.

모든 보조항은 물체/쌍/관계의 유효 항목 평균 후 유효 Sample 평균을 사용한다. 유효 항목 또는 Sample이 0개이면 해당 Loss는 Gradient Graph에 연결된 0이며 0으로 나누지 않는다. Status 단계의 전체 Loss는 L_D+λ_T L_T+Σλ_aux L_aux로 계산하고, Target이 없는 Sample을 가짜 No-op으로 대체하지 않는다.

평가에서는 실제 사용 분포를 따로 보고하고, Class별 Recall과 Macro-F1을 기록한다. 초기 기능 Gate는 Macro-F1≥0.90, 유효 요청 수용≥95%, 각 Negative Class Recall≥95%로 둔다. Status의 ok가 기하 안전 승인을 대체하지 않는다.

## 14. Evaluation Protocol

### 14.1 Prediction과 Source Identity

추론은 RGB+Text만 사용한다. Objectness≥.5 Query만 남기며 예측 개수가 GT와 다르면 Count 실패다. Source 예측으로 Hungarian 대응을 계산하고, 각 대응이 중심 오차≤10 mm 및 회전 Envelope IoU≥0.5인지 검사한다. 대응에 Target 위치를 사용하지 않는다. 이 대응을 Target 평가에 그대로 쓴다.

Count·대응 실패는 전체 Scene 성공 실패다. Source/Target P95는 대응이 확인된 물체에서 계산하되 대응 Coverage와 실패 Scene 수를 함께 보고한다. Detection 실패를 제외한 성공률만 최종 성능으로 제시하지 않는다.

### 14.2 기하와 지시

Target 안전 평가는 **GT a,b + 예측 Target Pose**로 한다. 모든 물체 쌍의 Polygon 최단거리≥10 mm, 테이블 경계 여유≥10 mm, 겹침 면적≤1e-12 m²를 요구한다. 거리 허용오차는 0.1 mm다. 이 평가는 Oracle Size 기반 학습 결과 평가이며 실행 안전 검사와 다르다.

선택 물체만 y 내림차순으로 정렬하고 Row 내 max y−min y≤10 mm가 되도록 기존 Greedy Row 판정을 사용한다. Row 대표 y는 min/max의 중점이며 구성원 오차≤5 mm다. Row 안은 x 오름차순, 전체는 위→아래다.

- 요청 방향 오차≤5°.
- 인접 수평 Edge Gap과 Row 띠 Gap은 요청값±5 mm(+수치 허용오차).
- 색상별 Row 순수성, 같은 색상 Row의 연속성, 그룹 사이 2g.
- 크기 순서는 외접 면적이며 인접 역전이 5%를 넘지 않아야 한다. 색상 그룹이 있으면 그룹 내부에서 검사한다.
- 선택 물체 Envelope가 요청 영역에 포함되어야 한다.
- Partial의 고정 물체는 GT Source 대비 위치≤3 mm, 방향≤3°.
- No-op은 모든 물체의 Keep 조건 및 해당 지시·안전 조건을 함께 검사한다. Keep-all에는 불필요한 Row 조건을 적용하지 않는다.

GT Dataset Validator는 고정 Pose의 **정확 일치**를 요구한다. NN 평가의 3 mm/3° 허용을 Dataset 생성에 적용하지 않는다. Source Asset/라벨 불변성을 검사하는 M5를 예측 결과에 그대로 넣지 말고, 같은 기하·의미 Predicate를 사용하는 Prediction Adapter를 구현한다.

**Partial Refinement 평가 범위:** 현재 입력 Scene에서 확정된 선택 집합 S에만 방향·Row·크기 순서·색상 그룹·요청 간격·요청 영역을 적용한다. 고정 집합 K에는 Keep를 적용하고, 전체 S∪K에는 물체 보존과 안전 기하를 적용한다. 예측 Target에서 “the leftmost object” 대상을 다시 골라 같은 Sample의 평가 대상을 바꾸지 않는다. 다음 Turn에서 새 지시를 해석할 때만 그 Turn의 Source로 다시 grounding한다. 고정 물체가 산개한 채 남아 있다는 이유만으로 Partial을 실패 처리하거나, 전체 장면의 Full Knolling을 요구하지 않는다.

Partial Task Success는 위 의미·보존·안전 조건의 동시 충족으로 판정한다. 선택 물체가 모두 움직여야 하는 것은 아니며, 이동량이 작다는 이유만으로 실패시키지도 않는다. Teacher Target과의 좌표 일치 및 J_move는 별도 모방·이동 효율 지표다. 6.6절의 유한 후보 집합 최소 이동은 Teacher 선택 기준이며 NN의 Task Success에 동일 후보나 동일 좌표를 강요하는 조건이 아니다. 의미·안전 검사를 통과하지 못한 예측을 작은 이동 비용 때문에 성공으로 인정하지 않는다.

자동 성공과 학습 사례의 품질도 분리한다. |S|=1인 사례는 대상 선택·재배치/방향 수정·고정 보존을 가르치지만 선택 물체 사이 정렬·간격 학습의 근거는 아니다. |S|≥2의 다중 물체 수정, 산개 Source와 이미 정돈된 Source를 나누어 성공률과 표본 수를 보고한다. 정돈된 Source 사례가 없으면 해당 능력은 미평가로 표기한다. 수동 QC의 Partial 정돈감은 선택 집합의 요청 결과를 기준으로 하고, 고정 물체 보존과 불필요한 이동은 별도로 평가한다. 전체 장면의 정돈감 개선을 필수 조건으로 두거나, 기하적 유효성만으로 좋은 학습 사례라고 판정하지 않는다.

### 14.3 보고 지표

| 지표 | 정의 |
|---|---|
| Count accuracy | 예측 개수 정확 Scene / 전체 |
| Source errors | 위치 Euclidean mm, 크기 max(|Δa|,|Δb|) mm, π 주기 각도 °의 Median/P95 |
| Target errors | Source 대응을 고정한 목표 위치·각도 Median/P95 |
| Raw Geometry Success | Count/Source 대응 성공 AND 전체 간격·충돌·테이블 경계 통과 |
| SSR | Count/대응/Geometry/요청 영역/방향/순서/그룹/간격/Keep를 모두 통과한 Scene 비율 |
| 조건별 성능 | N, 형태/크기, 색상 구성, 방향, 간격, 영역, 영어 지시 유형, Full/Partial/No-op별 SSR와 n |
| Language sensitivity | 동일 RGB의 명령 Pair 각각 성공, 교차 지시 조건을 실제로 구분하는지 |
| Keep violation | 고정 물체 하나라도 기준을 넘은 Partial Scene 비율 |
| No-op | 전체 물체 보존+해당 지시+Geometry 성공률 |
| Refinement diagnostic | 1/2/3단계별 오차·기하 위반; 최종 단계가 반드시 각 Scene에서 단조 개선될 필요는 없음 |

실제 교차 명령 차이가 없는 Pair는 Language sensitivity 분모에서 제외하고 개수를 명시한다. 최종 Test의 보고에는 Family 단위 Bootstrap 신뢰구간 또는 성공 수/전체 수를 함께 제시한다. 100 Scene QA의 수동 검토는 성능 Test를 대신하지 않는다.

Baseline은 Source 그대로 유지, 고정 기본 지시를 넣은 동일 Neural Model, Teacher Reference를 사용한다. Teacher는 GT 상태와 구조화된 지시를 사용하는 Oracle Reference이므로 RGB-only 모델과 동등한 입력 조건의 경쟁 모델로 주장하지 않는다.

## 15. Simulation과 실제 실행 확장

학습 후 첫 Simulation은 물체를 예측 Target으로 이동시킨 배치 검증이다. 이는 Manipulation 성공이 아니다. 다음 단계에서 물리 Simulator의 고정 grasp/lift/place Primitive로 실행하고 낙하·충돌·도달 불가를 측정한다. 초기 물체 이동 순서는 관측 Source의 결정적 순서로 처리하고, 필요하면 임시 Buffer를 사용하는 Controller를 별도 검증한다. Controller는 NN 최종 Target을 바꾸지 않는다.

동일 평면 상에서 Source/Target이 겹치는 순서 의존성과 Grasp 가능성은 최종 배치의 충돌 없음만으로 보장되지 않는다. Execution 실패는 Layout 실패와 구분해서 기록한다. 3-turn 평가는 실제 실행/시뮬레이션 결과를 재관측한 이미지와 다음 명시적 지시를 입력하며, 중간에 Teacher 정답으로 Scene을 Reset하지 않는다. 하나의 Episode에서 모든 Turn이 성공해야 Episode 성공이다.

추론에서는 GT 크기가 없으므로 예측 Envelope를 보수적으로 확장해 실행 전 검사한다. 별도 Calibration Validation에서 올바른 Detection에 한해 Source 예측 Envelope와 실제 Envelope의 Hausdorff 오차를 측정하고, Family별 최대 오차의 99-percentile에 1 mm를 더한 값을 인식 여유 δ_perc의 초기 추정치로 사용한다. 보정 오차 δ_calib와 정상 Grasp/Place 오차 δ_exec를 합산해 예측 Footprint를 팽창시킨다. 물체 수·도메인·카메라 조건이 바뀌면 다시 보정한다.

이 경험적 여유는 누락 물체·실패 Grasp·OOD에 대한 확률적 안전 보장이 아니다. 팽창 Footprint가 공통 최소 간격·테이블 경계 조건을 통과하지 못하면 실행을 중단하고 재관측하며 좌표를 Planner로 수정하지 않는다. 자연어의 선택 집합·요청 영역을 GT 의미 필드로 검사하는 것은 Offline Evaluation에서만 가능하다. 실제 추론의 Safety Validator에 숨은 GT 의미나 별도 Parser를 제공했다고 가정하지 않는다.

현실 로봇 도입의 초기 요청 간격은 30 mm부터 검증하고, 도달성·Grasp 공간·테이블 Calibration·Emergency Stop을 별도로 확인한다. 로봇 하드웨어와 물리 Simulator 선정은 현재 M0–M8/Neural Pilot의 선행 조건이 아니다.

## 16. Pilot 재검증과 진행 기준

### 16.1 반드시 재검증할 것

v1.3 Teacher로 기존 130 Sample을 고정 입력 재평가하고 새/기존 Target 비교표를 만든다. 자동 검사는 새 Manifest의 모든 Sample에서 통과해야 한다. 기존 39개 Regression과 v1.3 신규 검사를 모두 실행하되, 테스트 수 자체보다 의미·기하·재현성 계약 보존을 기준으로 한다.

추가 Fixture는 균형/기존 분할 모두 필요할 때, N=8·수직·불균형 색상, Partial 중앙이 막힌 경우, 동일 의미 영어 표현, Object 배열 순서 변경, 동일 선택·고정 집합을 유지한 No-op 재적용, 위치 기반 참조의 재해석으로 대상이 유지/변경되는 경우, Legacy ±π/2 동등성과 고정/No-op Pose 정확 복사, Legacy 정책 재현, 정책 혼합 거부를 포함한다. 위치 참조 대상이 바뀐 경우에는 새 의미의 충족 여부를 검사하며 No-op을 강제하지 않는다. 변경 없는 Source 파일의 해시와 기존 Split은 유지되어야 한다. 이번 문서 보정만으로 추가 Fixture가 모두 구현·통과했다고 표시하지 않는다.

전체 130 Sample을 자연어+Source+Target으로 사람이 검토한다. QC Overlay가 있는 이미지와 없는 이미지를 함께 제공한다. 정돈감/명령 충족/Partial 이동 자연스러움/관측 가능성을 각각 평가한다. Keep-all에는 정돈감 개선을 요구하지 않는다. Contact Sheet 9쌍 확인은 전수 검토가 아니다.

### 16.2 QA Gate

- 저장되는 v1.3 Sample의 기하·Identity·Instruction·좌표·Mask·Split 오류 0.
- 반복·배열 순서·의미 동일 문장·동일 선택/고정 집합 및 요청 조건의 No-op 일관성 100%. 위치 기반 참조의 재해석은 6.2절 기준으로 별도 판정한다.
- 기존 100 Source 보존; 변경 Target 수와 재렌더 수를 명시.
- 130개 전수 사람 검토 완료. 100 Primary 중 적용 가능한 품질 종합 점수 4/5 이상이 최소 90개; 추가 30 Sample도 개별 검토.
- 2/5 이하의 명백한 문제는 원인 미해결 상태로 남기지 않는다. 학습 허용 Manifest에는 하드 계약 통과 및 사람의 acceptable 판정을 받은 Sample만 포함한다.
- 반복되는 조건별 품질 결함은 좋은 사례만 다시 뽑아 숨기지 않고 Teacher 규칙에서 수정한다. 결함이 없다면 새 미적 최적화·새 Layout Family 실험을 추가하지 않는다.
- v1.3에서 생성 실패한 고정 입력은 Legacy 유효 자료로 보존하고 새 정책 완료율에서 실패로 보고한다. 전체 130 Migration이 성공하지 않으면 완전 Migration 완료라고 표시하지 않는다. 학습용 부분집합을 쓰려면 제외 목록과 조건별 Coverage를 명시한다.

J_shape는 더 좋은 미감을 보장하지 않으므로 v1.2/v1.3를 같은 Source·Instruction으로 비교한다. Partial은 동일한 후보 범위 내 이동 비용이 감소했는지 함께 보고하되, 14.2절에 따라 선택 집합의 정돈과 고정 보존을 평가한다. 산개한 고정 물체를 그대로 둔 유효 결과에 전체 장면의 정돈감 개선을 요구하지 않는다. 적용 가능한 품질이 규칙적으로 악화되면 Teacher 선택 규칙의 단일 수정만 재검토하고, NN Loss로 잘못된 정답을 보상하지 않는다.

### 16.3 Neural 구현으로 넘어가는 Gate

QA Gate와 단일 정책 Manifest가 준비되면 Dataset Adapter/Model/Matching/Loss 구현을 시작할 수 있다. S0는 검토된 Train Sample로 수행한다. 100 Scene만으로 S2 일반화 성능을 주장하지 않는다. CLI 100 제한, Chunk/Resume, 확장 Split Manifest를 해결한 뒤 계획량 데이터로 S1/S2를 진행한다.

[2026-11-03 Final Dataset Quality Review](final-dataset-quality-review.md)는 영어 130개 전체의 Assistant 검토와 자동 감사를 구분하여 기록한다. S0 적합성 권고는 사람의 품질 승인과 별개이며, 기존 Human Review Gate와 Training Approval Manifest를 대체하지 않는다. Minor 문체·배치 선호만으로 정답을 재생성하지 않는다.

초기 목표는 2–4개 물체의 올바른 대응과 목표 배치를 Detection+Target만으로 학습하는 것이다. 이 단계가 실패하면 순서대로 입력/단위/정답 연결 → Detection → 텍스트 조건 반응 → Refinement Gradient → 데이터 Coverage를 진단한다. S0/S1 실패 상태에서 Collision/Relation/Status를 동시에 켜지 않는다.

## 17. v1.2에서 v1.3으로 바뀐 내용

| 영역 | v1.2 / 현재 구현 | v1.3 |
|---|---|---|
| 프로젝트 상태 | 계획 중심; 이후 M0–M8 완료 | 완료 결과와 미구현 Neural/Interactive 범위 통합 |
| Teacher 후보 | k=1..4, 순차 Chunk | k=1..N_s, 순차+균형, 중복 제거 |
| Full 선택 | 행 수→면적→폭 | J_shape→면적→행 수 |
| Partial 선택 | 영역 중심 가까운 Translation | J_move 우선, 기존 Row 후보 포함 |
| 기하/색상/No-op | v1.2 수정 계약 | 유지 |
| 데이터 처리 | Seed로 전체 생성 | 원본 보존, 고정 Source Migration, 변경 Target만 렌더 |
| Schema | 1.2 | 구조 1.2 유지, 정책·Migration Sidecar 추가 |
| Legacy 테스트 | 현재 Teacher로 재생성 비교 | 정책별 재현 검사, 기존 독립 감사 유지 |
| 언어 | Legacy 한·영 제한 문법 | 활성 데이터·사용자 예시·평가는 영어 전용; 다국어는 선택적 후속 작업; 의미 불변, Counterfactual 교차 검증 |
| 입력 길이 | 앞선 NN 설계의 128 Token | silent truncation 금지, 최대 512/dynamic padding |
| NN 핵심 | Object Decoder + 3-stage Arrangement | 유지; 미정 세부값을 10절 기본값으로 고정 |
| 초기 Loss | 다수 목적 동시 적용 계획 | Detection + Target만 |
| Matching | Source 기반 Identity | 유지, Cost·단계별 재사용 명시 |
| Loss 척도 | 개별 Weight 조정 위험 | meter 복원 후 10 mm/5 mm/10° 고정 정규화 |
| 보조 Loss | 초기부터 사용 | 실패 조건 충족 시 하나씩 추가 |
| Status | 전체 기능 계획 | Positive Pilot에서 비활성; Negative 학습 단계 분리 |
| Evaluation | 기하·지시·보존 | 유지; Dataset exact와 Prediction tolerance 분리, Coverage·정책·실행 지표 명시 |

이 표 밖의 핵심 범위, 물체 수·치수, 좌표, 간격, 안전 조건, 색상 순수 Row, Source–Target Identity, Split 원칙은 유지한다. v1.3의 명시 규칙과 과거 문서가 충돌하면 v1.3은 이후 구현에 적용하고, 과거 산출물의 provenance는 그대로 남긴다.

## 18. 최소 수정 체크리스트

### A. 현재 Generator에서 필요한 변경

- [ ] M4 정책을 `legacy_v1_2`와 `balanced_rows_v1.3`으로 명시적으로 구분한다.
- [ ] k 확장·두 분할·중복 제거·J_shape·J_move·결정적 Tie-break를 구현한다.
- [ ] Partial 기존 Row 후보와 전체 후보의 이동 우선 비교를 반영한다.
- [ ] No-op, 색상 순수성, 고정 Pose, 실제 크기·방향 계산, 실패 상태 구분을 보존한다.
- [ ] M3 Row 띠 표현을 명확히 하고 구조화 의미·숫자가 변하지 않았는지 확인한다.
- [ ] M7에 고정 입력 Migration 및 변경 Target만 갱신하는 경로를 추가한다.
- [ ] M8에 정책·결정성·Counterfactual·품질/이동량 비교 보고를 추가한다.
- [ ] Legacy 재현 검사와 새 정책 검사를 분리하고 기존 감사 기준을 낮추지 않는다.

### B. 기존 100 Scene에서 다시 할 일

- [ ] 원본 Manifest/Source/Asset/Split 보존을 확인한다.
- [ ] 130 Sample의 Teacher 차이를 계산하고 변경 목록을 작성한다.
- [ ] 변경 Target만 렌더하고 관련 Annotation/Relation/Hash/QC를 일괄 갱신한다.
- [ ] 변경 없는 이미지도 새 Manifest 전체 감사에는 포함한다.
- [ ] 30 Counterfactual Pair의 교차 지시 조건을 검증한다.
- [ ] 130 Sample 전수 사람 검토와 조건별 품질 분포를 기록한다.
- [ ] 기존 채택률과 Migration 성공률을 별도 보고한다.

### C. Neural 구현에서만 필요한 변경

- [ ] 기존 Annotation을 RGB+Text 입력과 감독 라벨로 분리하는 Adapter를 만든다.
- [ ] Token 길이·좌표·각도·Size Head·Query Padding/Mask를 검증한다.
- [ ] 4-layer Object Decoder와 3-stage Arrangement, 같은 Query Identity를 구현한다.
- [ ] Source-only Hungarian 한 번, 같은 대응으로 모든 Target 감독을 구현한다.
- [ ] 초기 L_D+L_T와 고정 Scale/Weight, 조건부 보조 Loss 스위치를 구현한다.
- [ ] GT 크기·선택·잠금이 추론 입력에 들어가지 않는지 확인한다.
- [ ] Prediction Adapter와 Raw Geometry/SSR/Keep/언어 반응 지표를 구현한다.
- [ ] S0→S1→S2 순서로 Gate를 적용하고, 보조 Loss는 진단 후 추가한다.

연속 수정 데이터, Negative Status 데이터, 로봇 실행은 C의 기본 Neural Pilot 통과 이후 범위다. 이미 완료된 M0–M8 전체를 다시 작성할 필요는 없다.

## 19. 문서 적용 완료 조건

이 Revision 자체의 산출물은 통합 명세서, 17절 변경표, 18절 최소 수정 체크리스트, 16절 재검증 Gate다. 이후 구현 완료를 주장하려면 새 Teacher 정책·Migration·전수 QA 결과가 필요하고, Neural 성공을 주장하려면 별도의 고정 Test 결과가 필요하다. 현재 승인된 작업 범위에서는 문서만 갱신하며 코드·Dataset·Git 이력·학습 상태는 변경하지 않는다.
