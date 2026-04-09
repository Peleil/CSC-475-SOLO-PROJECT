# MIR 프로젝트 계획: Beat Tracking and Tempo Estimation

> **샘플 보고서 정렬 모드:** CSC475 샘플 진행 보고와 동일하게 **GTZAN Tempo-Beat**와 **GiantSteps**를 쓰고, 알고리즘은 **DSP · autocorr(OSS+ACF) · madmom(DBN) · 1D-StateSpace** 네 가지를 비교하는 것을 목표로 한다.

### 현재 진행 상황 (업데이트)

| 항목 | 상태 |
|------|------|
| **데이터** | `gtzan_genre`(mirdata 레이아웃) + GiantSteps 오디오·tempo annotation 확보 |
| **방법 ① DSP** | `scripts/eval_common.py` — librosa `beat_track` 기반 global tempo |
| **방법 ② autocorr** | `scripts/autocorr_tempo.py` — OSS + ACF + pulse (Percival–Tzanetakis 계열) |
| **방법 ③ madmom** | RNN + DBN, 비트 시각 및 `tempo_from_beat_times` |
| **방법 ④ 1DSS** | `jump_reward_inference.joint_tracker.joint_inference` (BeatNet + jump-reward) |
| **템포 평가** | `scripts/run_tempo_eval.py` — 출력 **4파일**: `results/tempo_dsp.csv`, `tempo_madmom.csv`, `tempo_autocorr.csv`, `tempo_1dss.csv` (GiantSteps + GTZAN, mirdata 실행 불필요) |
| **비트 평가** | `scripts/run_beat_eval.py` — 출력 **2파일** (`beat_madmom.csv`, `beat_1dss.csv`), **GTZAN만** (GiantSteps는 비트 GT 없음) |
| **공통 로더** | `scripts/eval_dataset.py` — 경로 매칭·제외 JSON |
| **Conda** | **csc475-dsp-madmom** + `requirements-dsp-madmom.txt` / **csc475-1dss** + `requirements-1dss.txt` (`environment-*.yml` 참고) |
| **보고서 LaTeX** | 미작성 |

**비트 GT:** GTZAN만 `gtzan_tempo_beat-main/beats/*.beats`. GiantSteps는 비트 평가 스크립트에서 **다루지 않음**.

---

## 0) 1DSS 개선 시도 로그 (요약)

아래는 이번 사이클에서 실제로 수행한 1DSS 개선/검증 시도 기록이다.

1. **기초 비교/진단 추가**
   - `scripts/compare_beat_methods_plot.py`로 DSP·madmom·1DSS 비트 시각을 같은 트랙에서 직접 비교.
   - GTZAN에서 1DSS 추정 비트 간격(IBI)과 타 방법 템포 관계를 빠르게 점검.

2. **IBI 기반 보정 가설 검증**
   - GTZAN 장르별로 `IBI/tempo ratio`를 계산해 규칙성 확인 (`scripts/gtzan_genre_ibi_tempo_ratios.py` 작성).
   - 결과: 장르별로 강한 일관 규칙은 없고, 단일 전역 스케일(예: x2.2)도 안정적 개선이 아님.

3. **복잡한 필터/스케일 파이프라인은 롤백**
   - per-track GT 기반 스케일 선택은 정보 누수(leakage)로 판단.
   - filter-1dss, raw/filtered/scaled 분기 코드 및 관련 산출물/그림 삭제 후 평가 스크립트 단순화.

4. **논문(2111.00704v2) 재현 시도**
   - 목표 지표를 tempo 정확도 대신 **GTZAN beat F-measure(0.07s 창)**로 맞춤.
   - `run_beat_eval.py`에 재현용 옵션(요약 JSON, 1DSS 원본 경로 입력 옵션) 추가해 비교 기반 정리.
   - 별도 paper env/스크립트로 v0.0.6 기반 재현을 시도했으나, Windows 의존성 충돌(numba/numpy/BeatNet/pyaudio)로 안정 재현 실패.
   - 사용자 요청에 따라 paper 전용 env/스크립트/결과물은 정리(삭제)함.

5. **현재 결론 (이번 라운드)**
   - 1DSS 개선을 위한 단순 스케일/필터 접근은 근본 해결이 아님.
   - 현 코드베이스는 기본 4방법 비교(DSP/autocorr/madmom/1DSS) 중심으로 유지하고,
     1DSS는 한계와 실패 패턴을 보고서에 명시하는 방향이 현실적.

---

## 1) 프로젝트 요약

이 프로젝트는 `papers/`의 선행연구와 재현 가능한 코드 실험을 바탕으로 **beat tracking**과 **tempo estimation**을 위한 Music Information Retrieval (MIR) 파이프라인을 구축·평가한다.

### 샘플과 맞추는 핵심 목표 (Must-have)

- **데이터셋:** GTZAN Tempo-Beat (다장르, tempo + beat), GiantSteps (EDM, tempo 중심).
- **비교 방법 4가지:** M-DSP, M-autocorr, M-madmom, M-1DSS.
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
| `run_tempo_eval.py` | DSP, madmom, autocorr, 1dss 템포 vs GT BPM | `tempo_dsp.csv`, `tempo_madmom.csv`, `tempo_autocorr.csv`, `tempo_1dss.csv` |
| `run_beat_eval.py` | madmom, 1dss 추정 비트 vs GT (**GTZAN만**) | `beat_madmom.csv`, `beat_1dss.csv` |

`run_tempo_eval` / `run_beat_eval` 공통: `--data-home`, `--sample-size`, `--seed`, `--exclude-invalid-json`. `--methods` 로 dsp/madmom/autocorr/1dss(템포) 또는 madmom/1dss(비트) 선택.

### D. 환경

- **csc475-dsp-madmom:** `requirements-dsp-madmom.txt` — DSP + madmom + autocorr. `run_tempo_eval --methods dsp,madmom,autocorr`, `run_beat_eval --methods madmom`.
- **csc475-1dss:** `requirements-1dss.txt` — `jump-reward-inference` 스택. `run_tempo_eval --methods 1dss`, `run_beat_eval --methods 1dss`.

---

## 3) 실행 체크리스트

### Phase A: 데이터 + 환경

- [x] GTZAN / GiantSteps 경로·오디오
- [x] `requirements-dsp-madmom.txt` / `requirements-1dss.txt` + `environment-dsp-madmom.yml` / `environment-1dss.yml`
- [ ] 전처리·샘플레이트 보고서 문구 고정 (DSP 22.05 kHz, madmom 44.1 kHz)

### Phase B: 방법 네 가지

- [x] M-DSP — `eval_common.estimate_dsp_tempo`
- [x] M-autocorr — `autocorr_tempo.estimate_autocorr_tempo`
- [x] M-madmom — `eval_common.estimate_madmom_beats`
- [x] M-1DSS — `eval_common.estimate_1dss_beats_and_tempo`

### Phase C: 평가

- [x] Tempo CSV 4종 + Beat CSV 2종 (GTZAN만)
- [ ] 최종 요약 표·ISMIR LaTeX 초안

---

## 4) 2주 크런치 일정 (요약)

| 일 | 목표 |
|---|---|
| **1** | 두 conda env 설치; dsp-madmom에서 템포/비트(madmom), 1dss에서 `--methods 1dss` 각 5곡 스모크 |
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

---

## 7) 최신 실행 기록 기준 성능/시간 요약 (2026-04-08)

터미널 실행 로그(`run_all_1dss_evals.py`, `run_all_dsp_madmom_evals.py`)의 tqdm 경과시간을 기준으로 정리했다.  
주의: 아래 시간은 **progress bar elapsed 합산값**으로, 실제 wall-clock과 수 초~수 분 차이가 있을 수 있다.

### A. 실행별 소요시간 (로그 계산)

| 실행 | 구성 | 소요시간(로그 기준) |
|---|---|---:|
| `run_all_1dss_evals.py` step 1 | tempo `--methods 1dss` (GiantSteps 10:55 + GTZAN 03:46) | **00:14:41** |
| `run_all_1dss_evals.py` step 2 | beat `--methods 1dss` (GTZAN 03:45) | **00:03:45** |
| `run_all_1dss_evals.py` total | step1 + step2 | **00:18:26** |
| `run_all_dsp_madmom_evals.py` step 1 | tempo `--methods dsp` (10:36 + 03:23) | **00:13:59** |
| `run_all_dsp_madmom_evals.py` step 2 | tempo `--methods madmom` (3:23:25 + 1:32:33) | **04:55:58** |
| `run_all_dsp_madmom_evals.py` step 3 | tempo `--methods autocorr` (3:18:56 + 1:09:09) | **04:28:05** |
| `run_all_dsp_madmom_evals.py` step 4 | beat `--methods madmom` (GTZAN 1:15:29) | **01:15:29** |
| `run_all_dsp_madmom_evals.py` total | step1~4 합계 | **10:53:31** |
| 전체 6개 평가 총합 | 위 두 실행 total 합계 | **11:11:57** |

### B. 곡당 평균 실행시간 (비교용)

곡 수 기준:
- 템포 평가: GiantSteps 664 + GTZAN 998 = **1662곡**
- 비트 평가: GTZAN **998곡**

| Task | Method | 총 소요시간 | 곡당 평균 시간 |
|---|---|---:|---:|
| Tempo (1662곡) | **dsp** | 00:13:59 | **0.505 s/track** |
| Tempo (1662곡) | 1dss | 00:14:41 | 0.530 s/track |
| Tempo (1662곡) | autocorr | 04:28:05 | 9.678 s/track |
| Tempo (1662곡) | madmom | 04:55:58 | 10.685 s/track |
| Beat (998곡) | **1dss** | 00:03:45 | **0.226 s/track** |
| Beat (998곡) | madmom | 01:15:29 | 4.538 s/track |

해석:
- **템포 속도:** `dsp ≈ 1dss`가 매우 빠르고, `autocorr`/`madmom`은 훨씬 느림.
- **비트 속도:** `1dss`가 `madmom`보다 매우 빠름.
- 즉, 현재 결과는 **정확도는 madmom 우세 / 속도는 dsp·1dss 우세**의 트레이드오프 구조.

### C. 템포 평가 결과 비교 (4 methods)

출처: `results/tempo_summary_dsp.json`, `tempo_summary_autocorr.json`, `tempo_summary_madmom.json`, `tempo_summary_1dss.json`  
(`n_with_metrics=1659`, `skip_bad_gt=3` 공통)

| Method | mean_error_pct (↓) | mean_mae_bpm (↓) | ACC1 (↑) | ACC2 (↑) |
|---|---:|---:|---:|---:|
| **madmom** | **11.05** | **13.35** | **0.828** | **0.946** |
| dsp | 21.84 | 25.88 | 0.622 | 0.810 |
| 1dss | 28.49 | 39.42 | 0.154 | 0.253 |
| autocorr | 38.35 | 43.76 | 0.371 | 0.766 |

해석:
- **madmom이 모든 템포 지표에서 1위**.
- **dsp가 2위권 베이스라인**.
- **autocorr는 ACC2 대비 ACC1이 낮아** octave/배수 계열 오차 경향이 큼.
- **1dss는 현재 설정 기준 템포 성능이 가장 낮음**.

### D. 비트 평가 결과 비교 (2 methods, GTZAN)

`beat_summary_madmom.json`은 현재 CSV와 불일치가 있어, 실제 CSV 재집계값을 함께 기록한다.

| Method | n(ok) | mean F (↑) | median F (↑) | std |
|---|---:|---:|---:|---:|
| **madmom** (`beat_madmom.csv` 재집계) | 998 | **0.8717** | **0.9915** | 0.2207 |
| 1dss (`beat_1dss.csv` 재집계) | 998 | 0.2480 | 0.2326 | 0.0942 |

해석:
- **비트 평가는 madmom이 압도적 우세**.
- **1dss는 평균 F가 낮아**, 현 파이프라인에서는 beat 품질 한계가 뚜렷함.

### E. 종합 결론 (현 결과 기준)

1. **전체 최고 성능:** `madmom` (tempo + beat 모두 우세)  
2. **차선 템포 베이스라인:** `dsp`  
3. **autocorr:** ACC2는 높지만 ACC1/MAE 약세  
4. **1dss:** tempo/beat 모두 낮아, 보고서에서 한계와 실패 패턴 명시 필요
