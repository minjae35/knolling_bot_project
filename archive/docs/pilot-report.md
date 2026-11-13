# Phase 1 Pilot 완료 보고서

2026-11-03. 새 저장소 `minjae35/knolling_bot_project`에서 M0–M8을 처음부터
구현했다. 이전 프로젝트 코드를 재사용하지 않았다. Neural Network, 학습,
로봇 시뮬레이션 및 50,000개 본 데이터셋 생성은 수행하지 않았다.

## 결과

- 서로 다른 Scene Family **100개**, Sample **130개**, 실제 Blender Render View **210개**.
- Primary: Full 50 / Partial 30 / No-op 20. No-op은 반복 지시 14 / 명시적 keep-all 6.
- 30개 Family에는 동일한 입력 이미지에 서로 다른 지시·정답을 가진 추가 Sample이 있다.
- Source 물체 495개, 장면당 2–8개. Box 166 / Chamfer 188 / Capsule 141.
- 색상: Red 168 / Green 150 / Blue 177. 사용한 Asset Family 114개.
- 지시 언어: 한국어 73 / 영어 57 Sample.
- 모든 Sample에서 대상 보존, ID, 기하, 잠금, 적용 가능한 지시 조건 및 렌더 검사를 통과했다.
- 감사 오류 **0건**. Annotation과 이미지/Mask/Render Metadata를 포함한 파일 760개를 검증했다.

Primary 제안은 101회 중 100개를 채택했다: 채택률 **99.01%**, 제외율 **0.99%**.
제외 1건의 사유는 `candidate_family_exhausted`이며 상태는 `generation_failure`다.
이를 불가능한 지시의 증거로 처리하지 않았다. 추가 지시는 30회 중 30개 채택했다.
렌더 검사로 제외한 Sample은 0개다. 이 수치는 **Teacher 데이터 생성 채택률**이며,
향후 Neural Network나 로봇의 성공률을 의미하지 않는다.

| 검사 | 실측 | 통과 기준 |
|---|---:|---:|
| Target 최소 Envelope 간격 | 10.1688 mm | 10 mm, 수치 허용오차 0.1 mm |
| Target 최소 Boundary Margin | 12.5813 mm | 10 mm, 수치 허용오차 0.1 mm |
| Mask–실제 Footprint 최소 IoU | 0.998774 | ≥0.98 |
| 최대 중심 투영 오차 | 0.636553 px | ≤1 px |
| 내부 RGB 색상 일치 최솟값 | 100% | ≥99% |

| Split | Scene Family | Sample | 사용 Asset Family |
|---|---:|---:|---:|
| Train | 70 | 89 | 32 |
| Validation | 10 | 13 | 26 |
| Development | 10 | 13 | 28 |
| Test | 10 | 15 | 28 |

Scene Family, Asset Family, 정규화된 평면 형상 및 ID를 제외한 동일 장면 해시를
검사했다. 서로 다른 Split 사이 중복은 없다. 색상·균일 확대축소·높이 변경만으로는
다른 Asset Family가 되지 않는다. 이는 작은 **geometry-OOD Pilot**이며, 각 Split의
크기로 신뢰도 높은 모델 성능을 추정할 수 있다는 뜻은 아니다.

## 구현·검증 이력

각 행은 해당 시점의 검증 수치다. 최종 코드에서 실제 Pilot 검증까지 포함한
**39개 테스트를 모두 함께 실행하여 48.05초에 전부 통과**했다. 일반 빠른 테스트에서는 Blender,
전체 Builder, 실제 Pilot 검사가 환경변수에 따라 생략되므로 명시적으로 구분한다.

| Module / 수정 | Commit | Conventional Commit | 검증 |
|---|---|---|---|
| M0 | `0e9c66f` | `feat(dataset): add schemas and coordinate utilities` | 3 통과 |
| M5 | `82e60cb` | `feat(dataset): validate geometry and layouts` | 13 통과 |
| M1 | `6b9b869` | `feat(dataset): implement procedural assets` | 18 통과 |
| M2 | `a3e7ddc` | `feat(dataset): generate randomized scenes` | 19 통과; 100 Seed 검사 포함 |
| M3 | `99e8151` | `feat(dataset): generate language instructions` | 26 통과 |
| M4 | `f091444` | `feat(dataset): implement layout teacher` | 30 통과; 반복 No-op 및 탐색 실패 구분 |
| M6 | `f372f76` | `feat(dataset): add Blender rendering` | 기하/RGB 검사 통과; 추가 파일 해시 검사에서 메타데이터 차이 발견 |
| M6 수정 | `9fda1c3` | `fix(dataset): canonicalize Blender PNG metadata` | 실제 반복 렌더 포함 32 통과 |
| M7 | `9c4960e` | `feat(dataset): build annotated samples` | 실제 10-scene 통합 생성 포함 34 통과 |
| M8 | `47ca812` | `feat(dataset): enforce split integrity` | 실제 렌더·Builder 포함 38 통과 |
| 언어 수정 | `8cda9be` | `fix(dataset): correct Korean selection particles` | 빠른 검사 36 통과; 외부 데이터/렌더 검사 3개 별도 실행 |
| 최종 Pilot 검사 | 이 보고서와 함께 커밋 | `test(dataset): validate pilot dataset` | 실제 100-scene 전수 재감사 및 Annotation 재생성 포함 전체 39 통과 |

모든 변경은 `main`에 순서대로 Push하며 강제 Push나 이력 재작성은 하지 않는다.
M6의 추가 해시 검사 결과를 확인하기 전에 `f372f76`을 Push한 실수가 있었다.
픽셀은 동일했고 PNG의 생성 시각/렌더 시간만 달랐다. 후속 수정으로 메타데이터를
제거하고 실제 반복 렌더의 파일 해시까지 검증했다. 실패를 숨기거나 이력을 지우지 않았다.

한국어 부분 지시의 조사 오류는 `8cda9be`에서 수정했다. Pilot의 18개 문장만
갱신했으며, 수정 전후의 모든 의미 필드·좌표·물체·이미지 해시가 동일함을 확인했다.
Annotation 해시·Manifest·QC 보고서를 함께 갱신하고 전수 재감사했다.
`provenance.json`에는 렌더 Revision과 수정 후 Annotation Revision을 모두 기록했다.

## 산출물과 검토 방법

로컬 경로: `artifacts/pilot_v1_2/` (약 77.65 MB, QC 포함; Git 제외).

```text
asset_registry.json                 절차적 Asset 등록 정보
generation.json                     제안/채택/제외 이력 및 분포
provenance.json                     Seed, 도구/좌표 정보, 코드 Revision
manifest.jsonl                      130개 Annotation 경로와 SHA-256
{train,validation,development,test}.jsonl
audit.json, COMPLETE.json           최종 감사 결과와 Manifest 해시
scenes/<family>/source/             RGB, uint16 Instance Mask, Render Metadata
samples/<sample>/annotation.json    현재/목표 Pose, ID, 지시, 관계, 검증 결과
samples/<sample>/target/            No-op 외 목표 RGB/Mask/Metadata
qc/index.html                      전체 Sample을 볼 수 있는 오프라인 QC
qc/contact_sheet.png               대표 9개 입력/목표 장면
qc/summary.json                     자동 QC 수치
work/                              재현용 Render Job과 Blender 로그
```

`qc/index.html`을 브라우저에서 열고 Mode/Split 필터를 사용한다. 각 카드에는
한국어/영어 지시, 입력/목표 이미지, 일관된 Object ID, `[L]` 잠금 표시가 있다.
원본 Annotation과 Mask에도 연결된다. Mask는 0–8의 정수값이므로 일반 이미지
뷰어에서는 검게 보일 수 있다. No-op 목표는 정확히 동일한 Source Render를 재사용한다.

Assistant가 대표 Contact Sheet의 9쌍과 `scene_0006_primary`의 8-object 배치,
`scene_0075_primary`의 keep-all, `scene_0000_counterfactual`의 변경 지시 결과를
이미지로 확인했다. 이 표본에서는 물체 소실, 시각적 충돌, ID/잠금 표시 누락을
관찰하지 않았다. 한국어 조사 오류는 이 검토 중 발견하여 수정했다.
**130개 Sample 전체를 사람이 검토한 것은 아니다.** 사용자의 자연어 의도, 가독성,
배치 다양성 검토는 여전히 `pending_user_review`다. 자동 통과를 사람의 승인으로
기록하지 않았다. Keep-all은 안전한 기존 장면을 유지하므로 축 정렬이 없어도 정상이다.

최종 Manifest SHA-256:
`b8276caeffb0287eb738888c892d11ac508913cee514873564f5785d9bc7da27`

## 재검증 및 다음 범위

```sh
RUN_BLENDER_TESTS=1 RUN_BUILDER_TESTS=1 PILOT_DATASET=artifacts/pilot_v1_2 conda run -n knolling python -m pytest -q
```

Dataset 생성기의 **소규모 기능·정합성 검증은 완료**됐다. 50,000개 운영 준비까지
끝났다는 의미는 아니다. 현재 CLI는 100 Family까지만 허용하고, 렌더는 작은 Batch의
동기 방식이며 중단 지점 재개를 지원하지 않는다. 대량 생성 전 사용자의 시각/언어 QC,
Chunk/Resume 설계와 처리량·저장 공간 확인이 필요하다. 현재 분포는 3개 절차적 형상,
3색, 고정 BEV, 제한된 한·영 문법이며 자유로운 지시나 실제 자산 일반화를 보장하지 않는다.
Partial은 안전한 산개 장면의 일부를 수정하는 단일 단계이며 다중 대화 이력은 없다.
실제로 렌더한 비실행(ambiguous/unsupported/infeasible) 지시 데이터는 포함하지 않고
관련 상태 인터페이스만 테스트했다.

이 시점에서 구현·생성을 중단한다. 본 데이터셋 확대, Neural Pilot, 학습 및 로봇
시뮬레이션은 실행하지 않았다.
