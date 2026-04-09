Beat Tracking and Tempo Estimation Project

## Quick Start (2주 크런치용)

1) Conda 환경을 **둘** 둔다 (DSP+madmom 스택 / 1DSS 스택). pip 의존성은 `requirements-dsp-madmom.txt`와 `requirements-1dss.txt`로 나뉜다.

```bash
conda env create -f environment-dsp-madmom.yml
conda activate csc475-dsp-madmom
pip install -r requirements-dsp-madmom.txt

conda env create -f environment-1dss.yml
conda activate csc475-1dss
pip install -r requirements-1dss.txt
```

2) GTZAN-Genre (mirdata) — `gtzan_genre` 데이터셋 받기

- **mini** (약 100곡 + tempo/beat 어노테이션, 빠른 점검):  
  `python scripts/download_gtzan_genre_mirdata.py --version mini`
- **전체** — mirdata 기본은 UVic `opihi.cs.uvic.ca` 에서 `genres.tar.gz` 를 받는데, **방화벽/VPN/서버 문제로 타임아웃(WinError 10060 등)** 이 나면 아래 **Hugging Face 미러**를 쓰면 된다 (`marsyas/gtzan` 저장소의 `data/genres.tar.gz`):  
  `python scripts/download_gtzan_genre_mirdata.py --version default --source huggingface`
- 전체를 UVic 경로로 받아 볼 때(네트워크가 될 때만):  
  `python scripts/download_gtzan_genre_mirdata.py --version default --source mirdata`
- 기본 저장 위치: `dataset/mirdata_gtzan_genre` (`--data-home`로 변경 가능)
- 다시 받을 때: `--force`
- 검증에서 일부 track checksum 불일치가 나와도 기본은 **경고 후 계속 진행**하며, 목록을 `results/invalid_tracks_gtzan_genre.json`에 저장한다. (`--strict-validate`를 주면 실패 처리)

3) GiantSteps만 먼저 쓸 때 (GTZAN을 mirdata로 받는 것과 별개로 진행 가능)

- 약 664개 mp3는 JKU 백업 미러에서 받는다(원본 `audio_dl.sh`와 동일한 URL·MD5 검증).
  - **Windows:** `python -u scripts/download_giantsteps_audio.py --giantsteps-root dataset/giantsteps-tempo-dataset-master --workers 4`
  - **Linux/Mac:** 같은 저장소의 `audio_dl.sh`(bash+curl) 사용 가능.
- mp3는 `dataset/giantsteps-tempo-dataset-master/audio/` 아래에 두면, 스크립트가 `1030011.LOFI.bpm` ↔ `1030011.LOFI.mp3` 로 매칭한다.

---

## 실행 커맨드 정리

아래 명령은 **저장소(프로젝트) 루트 디렉터리**에서 실행한다. (`cd`로 해당 폴더로 이동한 뒤 `python scripts/...`)

### 어떤 Conda 환경에서 무엇을 돌리나

| 환경 | 용도 | `run_tempo_eval.py` | `run_beat_eval.py` |
|------|------|---------------------|-------------------|
| **csc475-dsp-madmom** | librosa DSP, madmom, autocorr | `--methods dsp`, `madmom`, `autocorr` (조합 가능) | `--methods madmom` |
| **csc475-1dss** | jump-reward-inference (1DSS / BeatNet) | `--methods 1dss` | `--methods 1dss` |

- **템포 평가**는 GiantSteps + GTZAN-Genre를 지원한다 (`--dataset`).
- **비트 평가**는 **GTZAN만** (GiantSteps 공개 세트에 비트 GT 없음).

### 템포: `scripts/run_tempo_eval.py` (`csc475-dsp-madmom` 또는 `csc475-1dss`)

```bash
# DSP + madmom + autocorr (dsp-madmom 환경)
conda activate csc475-dsp-madmom
python scripts/run_tempo_eval.py --dataset both --sample-size 50 --methods dsp,madmom,autocorr

# 1DSS만 (1dss 환경)
conda activate csc475-1dss
python scripts/run_tempo_eval.py --dataset both --sample-size 50 --methods 1dss
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `both` | `giantsteps` \| `gtzan_genre` \| `both` |
| `--methods` | `dsp,madmom` | 쉼표 구분: `dsp`, `madmom`, `1dss`, `autocorr` |
| `--sample-size` | 전체 | 무작위 샘플 개수 상한 (`None`이면 제한 없음) |
| `--seed` | `0` | 샘플 순서 시드 |
| `--giantsteps-root` | `dataset/giantsteps-tempo-dataset-master` | GiantSteps 루트 |
| `--data-home` | `dataset/mirdata_gtzan_genre` | GTZAN: 오디오 `gtzan_genre/...`, 라벨 `gtzan_tempo_beat-main/` |
| `--exclude-invalid-json` | `results/invalid_tracks_gtzan_genre.json` | 제외할 track id 목록 |
| `--sample-rate` | `44100` | mono 리샘플 Hz (모든 메서드 동일 입력) |
| `--sample-rate-dsp` | — | 지정 시 `--sample-rate` 대신 사용(호환용 이름) |
| `--output-dir` | `results` | CSV·요약 JSON 저장 폴더 |
| `--no-progress` | — | tqdm 진행 표시 끔 |

산출물 (선택한 `--methods`마다):

- `tempo_dsp.csv`, `tempo_madmom.csv`, `tempo_1dss.csv`, `tempo_autocorr.csv`
- 메서드별 요약: `tempo_summary_dsp.json`, `tempo_summary_madmom.json`, …

### 비트: `scripts/run_beat_eval.py` (`csc475-dsp-madmom` 또는 `csc475-1dss`)

```bash
conda activate csc475-dsp-madmom
python scripts/run_beat_eval.py --sample-size 50 --methods madmom

conda activate csc475-1dss
python scripts/run_beat_eval.py --sample-size 50 --methods 1dss
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--methods` | `madmom` | `madmom`, `1dss`, 또는 `madmom,1dss` |
| `--sample-size`, `--seed` | — | 템포 스크립트와 동일 의미 |
| `--data-home`, `--exclude-invalid-json` | 위와 동일 | GTZAN 비트·템포 라벨 경로 |
| `--sample-rate`, `--sample-rate-dsp` | `44100` | madmom / 1dss 공통 입력 Hz |
| `--output-dir` | `results` | |
| `--no-progress` | — | |

산출물: `beat_madmom.csv`, `beat_1dss.csv` (요청한 메서드만).

### 데이터 다운로드 스크립트 (환경)

- **GTZAN (mirdata 레이아웃):** 보통 `csc475-dsp-madmom`에서 `requests` 등으로 받기 (`download_gtzan_genre_mirdata.py`).
- **GiantSteps mp3:** 동일 환경에서 `download_giantsteps_audio.py`.

---

## 결과 파일 (기본 `results/`)

- 템포: `tempo_dsp.csv`, `tempo_madmom.csv`, `tempo_autocorr.csv`, `tempo_1dss.csv`, `tempo_summary_*.json`
- 비트: `beat_madmom.csv`, `beat_1dss.csv`
- 검증 경고 트랙: `invalid_tracks_gtzan_genre.json`

오디오가 없으면 CSV에 `status=missing_audio` 등으로 표시된다.

---

1. For the final report the text will be copied and formatted as an ISMIR paper using LaTex. 

2. Code for possible uses: 
https://github.com/mjhydri/1D-StateSpace
madmom · PyPI
https://github.com/marsyas/marsgeyas


3. Papers to refer (pdfs are also in /papers folder):
Gouyon, Fabien, et al. "An experimental comparison of audio tempo induction algorithms." IEEE Transactions on Audio, Speech, and Language Processing 14.5 (2006): 1832-1844.

Percival & Tzanetakis (2014) — `scripts/autocorr_tempo.py` 에서 참고한 OSS+ACF+pulse 정식 (`--methods autocorr`).

Böck, Sebastian, and Markus Schedl. "Enhanced beat tracking with context-aware neural networks." Proc. Int. Conf. Digital Audio Effects. 2011.

4. datasets (샘플 진행 보고와 동일 계획)

- **GTZAN Tempo-Beat** — beat + global tempo, 다장르.
- **GiantSteps** — EDM·tempo 중심; annotation과 오디오 경로는 서브폴더마다 다를 수 있음.

로컬에는 `/dataset` 아래에 깃헙 repo를 통째로 두어도 된다. 권장: 역할별로 폴더명을 구분하고, 오디오가 없으면 해당 데이터셋 README의 **다운로드 스크립트**로만 받기. 대용량 오디오는 필요 시 `.gitignore`에 두고 경로만 문서화.

5. 비교 방법 (계획)

1. **DSP** — librosa `beat_track` 템포  
2. **autocorr** — Percival–Tzanetakis류 OSS+ACF+pulse  
3. **madmom** (DBN beat tracking)  
4. **1D-StateSpace** — https://github.com/mjhydri/1D-StateSpace  

상세 일정·체크리스트는 `MIR_Project_Summary_Blueprint_Checklists.md` 참고.
