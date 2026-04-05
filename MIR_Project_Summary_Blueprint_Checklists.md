# MIR 프로젝트 계획: Beat Tracking and Tempo Estimation

> **샘플 보고서 정렬 모드:** CSC475 샘플 진행 보고와 동일하게 **GTZAN Tempo-Beat**와 **GiantSteps**를 쓰고, 알고리즘은 **autocorrelation(DSP) · madmom(DBN) · 1D-StateSpace** 세 가지를 비교하는 것을 목표로 한다.

### 현재 진행 상황 (업데이트)

| 항목 | 상태 |
|------|------|
| **데이터** | `gtzan_genre`(mirdata 레이아웃) + GiantSteps 오디오·tempo annotation 확보 |
| **방법 ① DSP** | `scripts/eval_common.py` — librosa `beat_track` 기반 global tempo |
| **방법 ② madmom** | RNN + DBN, 비트 시각 및 `tempo_from_beat_times` |
| **방법 ③ 1DSS** | `jump_reward_inference.joint_tracker.joint_inference` (BeatNet + jump-reward) |
| **템포 평가** | `scripts/run_tempo_eval.py` — 출력 **3파일**: `results/tempo_dsp.csv`, `tempo_madmom.csv`, `tempo_1dss.csv` (GiantSteps + GTZAN, mirdata 실행 불필요) |
| **비트 평가** | `scripts/run_beat_eval.py` — 출력 **2파일** (`beat_madmom.csv`, `beat_1dss.csv`), **GTZAN만** (GiantSteps는 비트 GT 없음) |
| **공통 로더** | `scripts/eval_dataset.py` — 경로 매칭·제외 JSON |
| **Conda** | **csc475-tempo** + `requirements-tempo.txt` / **csc475-beat** + `requirements-beat.txt` (`environment-*.yml` 참고) |
| **보고서 LaTeX** | 미작성 |

**비트 GT:** GTZAN만 `gtzan_tempo_beat-main/beats/*.beats`. GiantSteps는 비트 평가 스크립트에서 **다루지 않음**.

---

## 1) 프로젝트 요약

이 프로젝트는 `papers/`의 선행연구와 재현 가능한 코드 실험을 바탕으로 **beat tracking**과 **tempo estimation**을 위한 Music Information Retrieval (MIR) 파이프라인을 구축·평가한다.

### 샘플과 맞추는 핵심 목표 (Must-have)

- **데이터셋:** GTZAN Tempo-Beat (다장르, tempo + beat), GiantSteps (EDM, tempo 중심).
- **비교 방법 3가지:** M-DSP, M-madmom, M-1DSS.
- **평가:** Tempo — ACC1, ACC2, MAE, %오차. Beat — `mir_eval` F-measure (GT 있을 때만).

---

## 2) Blueprint (엔드투엔드)

### A. 연구 질문

- **RQ1:** classical(DSP) vs probabilistic(madmom) vs state-space(1D-SS)의 **tempo·beat** 성능 차이.
- **RQ2:** 장르(GTZAN) vs EDM(GiantSteps)에서 실패 패턴.

### B. 데이터/어노테이션

1. **GiantSteps:** `annotations_v2/tempo/*.bpm` + `audio/*.mp3` — `scripts/download_giantsteps_audio.py`.
2. **GTZAN:** `dataset/mirdata_gtzan_genre/` — `gtzan_genre/genres/{장르}/*.wav`, `gtzan_tempo_beat-main/tempo`, `beats`. 무결성 실패 ID는 `results/invalid_tracks_gtzan_genre.json`로 제외.

### C. 평가 프레임워크 (실행)

| 스크립트 | 내용 | 산출물 (기본 `results/`) |
|----------|------|--------------------------|
| `run_tempo_eval.py` | DSP, madmom, 1dss 템포 vs GT BPM | `tempo_dsp.csv`, `tempo_madmom.csv`, `tempo_1dss.csv` |
| `run_beat_eval.py` | madmom, 1dss 추정 비트 vs GT (**GTZAN만**) | `beat_madmom.csv`, `beat_1dss.csv` |

`run_beat_eval` 인자: `--data-home`, `--sample-size`, `--seed`, `--exclude-invalid-json`.

### D. 환경

- **csc475-tempo:** `requirements-tempo.txt` — DSP + madmom + 안정 numpy. `tempo_1dss.csv` 는 jump-reward 미설치 시 행마다 `1dss_fail`.
- **csc475-beat:** `requirements-beat.txt` — madmom + `jump-reward-inference` + PyAudio. **템포 스크립트를 여기서 실행**하면 1dss 열까지 채우기 쉽다 (의존성 충돌 시).

---

## 3) 실행 체크리스트

### Phase A: 데이터 + 환경

- [x] GTZAN / GiantSteps 경로·오디오
- [x] `requirements-tempo.txt` / `requirements-beat.txt` + `environment-tempo.yml` / `environment-beat.yml`
- [ ] 전처리·샘플레이트 보고서 문구 고정 (DSP 22.05 kHz, madmom 44.1 kHz)

### Phase B: 방법 세 개

- [x] M-DSP — `eval_common.estimate_dsp_tempo`
- [x] M-madmom — `eval_common.estimate_madmom_beats`
- [x] M-1DSS — `eval_common.estimate_1dss_beats_and_tempo`

### Phase C: 평가

- [x] Tempo CSV 3종 + Beat CSV 2종 (GTZAN만)
- [ ] 최종 요약 표·ISMIR LaTeX 초안

---

## 4) 2주 크런치 일정 (요약)

| 일 | 목표 |
|---|---|
| **1** | 두 conda env 설치; `run_tempo_eval` / `run_beat_eval` 각 5곡 스모크 |
| **2~4** | 전체 샘플 배치; 실패 트랙 정리 |
| **5~14** | 표·그림·LaTeX |

**구제 플랜:** 1DSS만 막히면 `tempo_1dss.csv` / `beat_1dss.csv` 를 제외하고 DSP·madmom만 보고서에 넣고 한계 명시.

---

## 5) 리스크

- **jump-reward-inference**와 **numpy/numba** 핀 충돌 → 두 env 분리.
- **GiantSteps** → 비트 평가 대상 아님 (비트 GT 없음).

---

## 6) Definition of Done

- [x] 템포·비트 평가 스크립트 분리 및 `results/` 산출물 규칙 문서화
- [ ] 두 데이터셋·세 방법으로 표 1장 이상
- [ ] `README` 실행 명령과 conda 이름 일치
