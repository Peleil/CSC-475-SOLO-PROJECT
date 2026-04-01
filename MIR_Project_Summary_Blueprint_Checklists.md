# MIR 프로젝트 계획: Beat Tracking and Tempo Estimation

> **샘플 보고서 정렬 모드:** CSC475 샘플 진행 보고와 동일하게 **GTZAN Tempo-Beat**와 **GiantSteps**를 쓰고, 알고리즘은 **autocorrelation(DSP) · madmom(DBN) · 1D-StateSpace** 세 가지를 비교하는 것을 목표로 한다.  
> **2주 크런치**가 필요하면 §4·§5를 우선 따르되, **방법 3개 + 데이터 2종**은 부담이 크므로 §1의 **우선순위**로 현실적으로 줄인다.

## 1) 프로젝트 요약

이 프로젝트는 `papers/`의 선행연구와 재현 가능한 코드 실험을 바탕으로 **beat tracking**과 **tempo estimation**을 위한 Music Information Retrieval (MIR) 파이프라인을 구축·평가한다.  
최종 결과물은 실행 가능한 코드/평가 산출물과 함께, **ISMIR 스타일 LaTeX 논문**으로 옮기기 쉬운 보고서 초안이다.

### 샘플과 맞추는 핵심 목표 (Must-have)

- **데이터셋 (샘플과 동일 계획):**
  - **GTZAN Tempo-Beat:** 다장르, beat + global tempo annotation.
  - **GiantSteps:** EDM 중심, tempo(및 사용 가능한 beat)로 샘플 보고서처럼 **템포 구간 층화 split** 등 보완 가능.
- **비교 방법 3가지 (고정):**
  1. **M-DSP / autocorrelation baseline** — Percival–Tzanetakis류 tempo + 필요 시 간단 beat 파이프라인.
  2. **M-madmom** — `RNNBeatProcessor` + `DBNBeatTrackingProcessor` 등 **madmom** 기반 tracker.
  3. **M-1DSS** — `mjhydri/1D-StateSpace` 저장소의 **1D state-space** beat/tempo 추적.
- **평가:** 샘플에 가깝게 가려면 `mir_eval` 기준 **ACC1, ACC2, MAE**(tempo), **F-measure**(beat, 예: 70 ms tolerance)를 우선. 시간이 없으면 **ACC1/ACC2 + F-measure + MAE**만 고정.
- **보고서:** 정량 표 + 실패 사례 몇 곡 + 샘플처럼 **통일된 전처리·동일 Evaluator 인터페이스**를 강조하면 설득력이 좋다.

### `dataset/`을 “깃헙 통째 복사”했을 때 권장 사항

통째 복제도 **가능**하지만, 아래를 하면 나중에 재현·제출·용량 관리가 쉽다.

| 방식 | 장점 | 비고 |
|------|------|------|
| **지금처럼 로컬에 클론/복사** | 오프라인 작업, 설정 단순 | `README`에 **각 서브폴더가 어떤 데이터셋인지**, 오디오 경로 규칙을 적어 둘 것. 대용량은 **Git에 안 올리고** `.gitignore` + 별도 백업. |
| **`git submodule`** (데이터용 repo만) | 버전 고정, 팀/교수에게 “어떤 스냅샷인지” 설명 용이 | 처음 한 번만 학습 비용. |
| **공식 스크립트로 오디오만 받기** | repo 메타데이터 vs 실제 wav/mp3 분리 | GiantSteps 등은 annotation만 있고 **오디오는 스크립트로 다운로드**인 경우가 많음 → `audio_dl.sh` 등 확인. |
| **심볼릭 링크** | 디스크 중복 감소 | Windows에서 권한/백업 정책 확인. |

**추천 정리:**  
1) `dataset/` 아래를 **`dataset/gtzan_tempo_beat/`**, **`dataset/giantsteps/`**처럼 **역할별 폴더명**으로 정리(이미 giantsteps가 있으면 그대로 두되 README에 한 줄 매핑).  
2) **오디오가 있는지** 첫날 스캔: 없으면 `missing_audio`처럼 평가에서 제외할지, 다운로드할지 결정.  
3) 샘플 보고서 수준으로 가려면 **동일 sample rate(예: 44.1 kHz), mono, 동일 정규화**를 세 방법 모두에 적용.

### 명시적으로 나중으로 미룰 수 있는 것 (시간 부족 시)

- Cemgil, Information Gain, Wilcoxon 등 **부가 metric·통계 검정** (샘플 수준은 목표로 두고, 마감 전에는 표·F-measure 중심으로 축소 가능).
- `marsyas` 딥 통합, extensive hyperparameter sweep.

### 권장 코드·참고

- **1D-StateSpace:** https://github.com/mjhydri/1D-StateSpace  
- **madmom** (PyPI)  
- **선행 논문:** Gouyon et al. (2006), Percival & Tzanetakis (2014), Böck & Schedl (2011)

---

## 2) Blueprint (엔드투엔드 프로젝트 설계)

### A. 연구 질문과 범위 (샘플에 맞춘 예시)

- **RQ1:** classical(autocorrelation) vs probabilistic(madmom) vs state-space(1D-SS)의 **tempo·beat 성능** 차이는?  
- **RQ2:** 장르(GTZAN) vs EDM(GiantSteps)에서 **실패 패턴**이 다른가?  
- **RQ3 (옵션):** 정확도–실행 시간 트레이드오프.

초기에 범위를 명확히 정의한다.

- Task 1: **Global tempo estimation**  
- Task 2: **Beat tracking**  
- **방법 3개:** M-DSP, M-madmom, M-1DSS — 입력은 가능하면 **`predict(audio_path)` 한 가지 인터페이스**로 통일(샘플 보고서의 Evaluator 패턴).

### B. 데이터/어노테이션 워크플로우

1. **우선순위 (현재):** GiantSteps만 진행 가능 — `annotations_v2/tempo/*.bpm` + 오디오(mp3)는 README/`audio_dl.sh`로 별도 확보 후 `dataset/giantsteps-tempo-dataset-master/audio/` 등에 배치. 스크립트: `scripts/run_giantsteps_quick_check.py`.  
2. **GTZAN Tempo-Beat:** 오디오는 나중에 받아도 됨. 준비되면 **오디오 경로 + beat 파일 + tempo(BPM)** 매칭표 작성.  
3. GiantSteps: **템포 구간별 층화 split**은 샘플처럼 하면 좋고, 시간이 없으면 **단순 70/15/15 한 번**만. (beat 정답은 이 데이터셋에 없으므로 **tempo 평가**를 먼저 완성.)  
3. 전처리: **44.1 kHz, mono, 정규화 정책 고정** (샘플 보고서와 동일하게 맞추기 쉬움).  
4. integrity: 여유 있으면 **MD5 또는 파일 크기**로 손상 파일 제거; 최소한 **로드 실패 트랙 로그**.

### C. 방법론 로드맵

- **M-DSP:** autocorrelation / onset 기반 tempo; beat는 librosa 또는 별도 간단 디코더.  
- **M-madmom:** RNN activation + DBN beat tracking; 출력을 **초 단위**로 통일해 `mir_eval`에 넣기.  
- **M-1DSS:** `1D-StateSpace` repo의 **README·환경 요구사항**(Python 버전, 의존성) 확인 후, 동일 오디오 입력으로 beat(및 제공 시 tempo) 추출.

각 방법마다 문서화: 입력 feature, 가정, hyperparameter, 실패 시 empty output 처리(저 BPM 등).

### D. 평가 프레임워크

- Tempo: **ACC1, ACC2**, **MAE**.  
- Beat: **F-measure** (tolerance 고정, 예: 70 ms).  
- 시간 남으면: Cemgil, Information Gain 등 샘플 보고서와 동일선상으로 확장.

분석: 전체 표 + (가능하면) 장르별·데이터셋별 소표; failure case 2~5곡.

### E. 실험 실행 계획

1. 단일 **Evaluator** 루프: 트랙마다 세 방법 `predict` → JSON/CSV 저장.  
2. 로그: split, 버전, seed, wall-clock time.  
3. clean 재현: 제출 전 **한 번** 전체 재실행 권장.

### F. 보고서 Blueprint (ISMIR)

Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion.

---

## 3) 실행 체크리스트 (압축)

### Phase A: 데이터 + 환경

- [ ] `dataset/` 아래 **GTZAN / GiantSteps** 위치와 오디오 유무 스캔  
- [ ] `requirements.txt` + (필요 시) **1D-StateSpace / madmom** 각각 요구 버전 메모  
- [ ] 전처리 한 함수로 통일 후 **샘플 5곡**으로 세 방법 모두 통과

### Phase B: 방법 세 개

- [ ] M-DSP 스크립트 또는 모듈  
- [ ] M-madmom 배치  
- [ ] M-1DSS 클론 경로 고정 + 실행 커맨드 문서화  

### Phase C: 평가

- [ ] `mir_eval`로 ACC1/ACC2/MAE/F-measure  
- [ ] 전체 test split batch → **하나의 요약 CSV**  
- [ ] 표 + 그림 + limitation

---

## 4) 2주 크런치 일정 (세 방법·두 데이터셋 전제)

시간이 촉박할 때 **구현 순서**만 지킨다: **① madmom (가장 표준)** → **② DSP** → **③ 1D-StateSpace (환경 이슈 가능)**.

| 일 | 목표 |
|---|---|
| **1** | 두 데이터셋 경로·오디오·annotation 매칭; 3곡으로 madmom만 end-to-end |
| **2** | 전처리 고정; DSP tempo(+beat 가능 시) 프로토타입 |
| **3** | 1D-StateSpace 설치·샘플 실행; 출력 형식 madmom과 동일하게 맞추기 |
| **4** | `mir_eval` 연결; 10곡으로 세 방법 수치 교차검증 |
| **5~7** | 전체 배치 1차; 실패 트랙 규칙화 |
| **8~10** | 표·그림·실패 사례·초안 |
| **11~14** | LaTeX 이전, 재현 1회, 제출 패키징 |

**구제 플랜:** 1D-StateSpace만 막히면 보고서에 **환경 제약 + madmom·DSP 이원 비교**로 일단 마감하고, 1D-SS는 **부록 또는 후속**으로 명시.

---

## 5) 리스크 관리 메모

- **1D-StateSpace:** Python/Cython·버전 핀ning 이슈 가능 → 초기에 고정.  
- **GiantSteps 오디오:** repo만 두면 annotation만 있을 수 있음 → 다운로드 스크립트 확인.  
- **세 방법 출력 단위:** 프레임 vs 초 혼동 시 metric 전부 틀어짐 → 초기에 `mir_eval` 한 곡 수동 대조.

---

## 6) Definition of Done

- [ ] **GTZAN + GiantSteps** 중 과제에서 요구하는 범위가 명확히 서술됨  
- [ ] **M-DSP, madmom, 1D-StateSpace** 세 방법이 **동일 전처리·동일 metric**으로 비교됨 (1D-SS 불가 시 구제 플랜과 함께 문서화)  
- [ ] 결과 CSV/JSON + 요약 표 + 실패 사례  
- [ ] `README`에 데이터 경로·실행 명령·환경(conda/python 버전) 명시
