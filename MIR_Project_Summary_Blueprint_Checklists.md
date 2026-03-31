# MIR 프로젝트 계획: Beat Tracking and Tempo Estimation

> **2주 크런치 모드:** 아래 계획은 “완성도보다 제출·발표에 필요한 최소선”을 우선한다. 장문 Blueprint는 유지하되, **실행 순서는 §4(2주 일정)**와 **§3 압축 체크리스트**를 따른다.

## 1) 프로젝트 요약

이 프로젝트는 `dataset/`의 데이터, `papers/`의 선행연구, 재현 가능한 코드 실험을 바탕으로 **beat tracking**과 **tempo estimation**을 위한 Music Information Retrieval (MIR) 파이프라인을 구축하고 평가하는 것을 목표로 한다.  
최종 결과물은 실행 가능한 코드/평가 산출물과 함께, **ISMIR 스타일 LaTeX 논문**으로 옮기기 쉬운 보고서 초안이다.

### 2주 기준으로 줄인 핵심 목표 (Must-have)
- **Global tempo (BPM)** 과 **beat 시점**을 같은 파이프라인에서 처리할 수 있게 한다.
- 방법은 **딱 2개만** 비교한다: (1) 직접 구현하는 가벼운 DSP baseline, (2) **toolkit baseline** (`madmom` 등). `1D-StateSpace`/신경망 직접 재현은 우선 버린다.
- 평가는 **tempo: 간단 tolerance 또는 MAE**, **beat: F-measure(고정 tolerance)** 정도로 고정해 반복 실험 시간을 줄인다.
- 보고서에는 **표 1장 + 실패 사례 2~3개**만 있어도 통과선을 노린다.

### 명시적으로 나중으로 미루는 것 (2주에서는 생략 권장)
- 장르별/tempo-bin별 전체 분해, Cemgil 등 부가 metric, cross-validation, `marsyas` 딥 통합, extensive hyperparameter sweep.

### 권장 baseline 방향 (README 기반)
- State-space / probabilistic tracking 아이디어 (`1D-StateSpace`)
- 기존 MIR 라이브러리 baseline (예: `madmom`)
- 추가 프레임워크 참고 (`marsyas` 관련 repo 링크)

### 프로젝트 프레이밍에 중요한 참고문헌
- Gouyon et al. (2006): tempo induction 알고리즘 비교 분석
- Percival and Tzanetakis (2014): autocorrelation/cross-correlation 기반 streamlined tempo estimation
- Boeck and Schedl (2011): context-aware neural-network beat tracking

---

## 2) Blueprint (엔드투엔드 프로젝트 설계)

## A. 연구 질문과 범위
- **RQ1:** 현재 데이터에서 toolkit vs 간단 DSP 중 tempo estimation이 더 나은 쪽은?
- **RQ2:** 동일 조건에서 beat tracking F-measure는 어떻게 다른가?
- **RQ3 (옵션):** 시간 남으면 장르 또는 BPM 구간 중 한 축만 짧게 코멘트.

초기에 범위를 명확히 정의한다.
- Task 1: **Global tempo estimation** (단일 BPM 출력이면 충분)
- Task 2: **Beat tracking** (beat timestamp 시퀀스)
- **방법 2개 고정:** M-DSP(또는 노트북 수준 pipeline) + M-toolkit(`madmom`).

## B. 데이터/어노테이션 워크플로우 (2주판: 최소 절차)
1. `dataset/`에서 **쓸 파일 목록 + annotation 경로**만 스프레드시트/텍스트로 고정한다.
2. beat / BPM 라벨 중 **하나라도 없으면** 해당 트랙은 첫날에 제외하거나, BPM만 beat에서 유도하는지 교수/명세 확인(반나절 안에 결정).
3. 전처리: **mono + 한 가지 sample rate만** 쓴다. loudness 정규화는 전곡 통일 또는 전부 끄기 중 하나.
4. split: **train 0도 가능**. 가능하면 **80/20 한 번**만; 어노테이션이 적으면 **전체 test**로만 수치 내고 limitation에 명시.

## C. 방법론 로드맵 (2주판)
- **M-DSP:** Percival–Tzanetakis류 **autocorrelation tempo** + (같은 onset envelope에서) **간단 beat 정렬**(DP 없이 주기 피킹 수준이면 첫 주에 막히면 `madmom` beat만 비교하고 tempo만 DSP로도 됨).
- **M-toolkit:** `madmom`으로 **tempo + beat 한 번에** 강한 baseline 확보.

각 방법마다 아래를 문서화한다.
- 입력 feature
- 모델/알고리즘 가정
- hyperparameter
- 실행 시간 복잡도 및 dependency

## D. 평가 프레임워크
Task별로 metric을 분리해 사용한다.

Tempo estimation:
- Accuracy1 / Accuracy2 계열 metric (정확 일치와 octave 관련 허용 오차 반영)
- 필요 시 절대 BPM 오차 (MAE)

Beat tracking:
- 허용 오차 window 기반 F-measure
- 선택적으로 continuity 기반 metric, Cemgil 계열 점수

분석 관점 (2주판):
- 전체 평균 표 1개 + 선택적 막대그래프 1개
- failure case **2~3곡**만 스크린샷/짧은 설명

## E. 실험 실행 계획
1. 단일 실험 실행 스크립트/노트북 구성
2. 각 실행 로그 기록
   - dataset split
   - 방법 및 hyperparameter
   - metric과 timestamp
3. 출력 저장
   - predicted beat 파일
   - predicted BPM 파일
   - 요약 CSV/JSON
4. 시간이 없으면 clean 재현은 **최종 제출 직전 하루**만 수행

## F. 보고서 Blueprint (ISMIR paper 전환 대비)
지금부터 LaTeX로 바로 옮길 수 있는 구조로 작성한다.
- **Abstract:** 문제 정의, 방법, 핵심 결과
- **Introduction:** MIR 맥락과 동기
- **Related Work:** 핵심 논문 3편 + 사용 baseline 도구 요약
- **Methodology:** 데이터, 전처리, 방법, metric
- **Experiments:** 실험 설정, split, 구현 세부
- **Results:** 정량 결과표 + 정성 오류 분석
- **Discussion:** 한계, 통찰, 향후 개선 방향
- **Conclusion:** 최종 결론과 최우수 방법

---

## 3) 2주 크런치 체크리스트 (짧게)

### Day 0~1 (반나절~1일): 환경 + 데이터
- [ ] `requirements.txt` 최소 패키지 + `madmom` 설치 확인
- [ ] `dataset/` 목록 + annotation 매칭 + **평가에 쓸 트랙 N 확정**
- [ ] 오디오 로드 함수 1개 + 리샘플링/mono 1줄 정책 고정

### Day 2~4: 방법 두 개 돌아가게 만들기
- [ ] M-DSP: tempo (필수) + 가능하면 beat (선택, 막히면 beat는 `madmom`만 비교)
- [ ] M-toolkit: `madmom`으로 tempo+beat 추론 스크립트 완성
- [ ] 출력 형식 통일: `track_id, bpm / beat_times[]` 저장

### Day 5~7: 평가 + 숫자 확보
- [ ] tempo metric 1종 + beat F-measure 1종만 고정
- [ ] 전 트랙 batch 실행 → **CSV 하나**로 합치기
- [ ] 표: 평균±표준편차 또는 단순 평균

### Day 8~10: 글 + 그림 + 제출
- [ ] Related work 1~2쪽 + 방법 1쪽 + 실험 1쪽 + 결과 1쪽 초안
- [ ] failure case 2~3곡 figure
- [ ] limitation(데이터 양, split, metric 단순화) 한 단락
- [ ] `README.md`에 “한 줄 실행” 명령만

---

### 참고: 여유 있을 때만 하는 확장 (구 4주 스타일)
- 폴더 구조 정리, 방법 3개째, 부가 metric, 장르별 분해, extensive 테스트 등은 시간 날 때 §2 전체 Blueprint를 참고해 확장한다.

---

## 4) 2주 일정 (크런치 기준)

가정: **매일 MIR에 쏟을 수 있는 시간이 대략 3~6시간** 수준. 더 짧으면 “짝수일만” 합쳐서 동일 블록으로 묶으면 된다.

### 제1주 — 파이프라인이 돈다 (숫자 전까지)
| 일 | 목표 |
|---|---|
| **1일차** | 환경, 데이터 목록, 3곡으로 `madmom` 추론·시각화 검증 |
| **2일차** | 전처리 고정 + DSP tempo 프로토타입 (노트북 OK) |
| **3일차** | DSP 출력 저장 포맷 통일 + `madmom` 배치 스크립트 |
| **4일차** | ground truth 로딩 검증 (시간 단위, 첫 beat 오프셋) — 여기서 버그 잡기 |
| **5일차** | metric 코드 붙이기: 곡 5~10곡 수동/샘플로 숫자 맞는지 확인 |
| **6~7일차** | 전체 트랙 1차 돌리기 + CSV; 막히는 곡은 로그 남기고 스킵 규칙 문서화 |

### 제2주 — 숫자 고정 + 보고서
| 일 | 목표 |
|---|---|
| **8일차** | 최종 표 확정; 필요 시 하이퍼 1개만 조정(반복 금지) |
| **9일차** | failure case 2~3개 피규어 + 짧은 해석 |
| **10일차** | 초안 완성 (서론/방법/실험/결과/한계) |
| **11일차** | ISMIR LaTeX로 옮기기 + 참고문헌 |
| **12~14일차** | 교정, 재실행 1회(환경 명시), 코드/결과 zip 정리 |

**시간이 정말 없을 때의 구제 플랜:** beat 비교는 잠시 접고 **tempo만** 표 + `madmom` 단일 baseline + limitation에 “beat는 후속 작업”이라고 명시. (과제 허용 범위는 확인.)

---

## 5) 리스크 관리 메모

- **Annotation 불일치 리스크:** 단위와 시간 정렬을 초기에 반드시 검증
- **Metric 혼동 리스크:** tempo metric과 beat metric을 명확히 분리
- **Scope 확장 리스크:** 약한 다수 방법보다 강한 2~3개 방법 우선
- **재현성 리스크:** seed, 버전, 실행 설정을 모두 기록

---

## 6) Definition of Done (2주판)

- [ ] **방법 2개**가 동일 트랙 집합에서 동일 metric으로 비교됨
- [ ] 결과가 **하나의 CSV + 하나의 요약 표**로 남음
- [ ] 보고서에 **정량 1표 + 실패 사례 2~3개** 이상
- [ ] `README`에 설치/실행에 필요한 최소 명령이 있음
- [ ] 데이터/annotation 한계와 생략한 분석은 본문에 짧게라도 명시됨

(시간이 남으면 원본 Definition of Done의 3방법·clean 재현·세부 분해로 확장.)
