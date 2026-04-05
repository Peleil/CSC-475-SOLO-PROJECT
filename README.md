Beat Tracking and Tempo Estimation Project

## Quick Start (2주 크런치용)

1) Conda 환경을 **둘** 둔다 (템포용 / 비트용). 루트 `requirements.txt`는 분기 안내만 있다.

```bash
conda env create -f environment-tempo.yml
conda activate csc475-tempo
pip install -r requirements-tempo.txt

conda env create -f environment-beat.yml
conda activate csc475-beat
pip install -r requirements-beat.txt
```

- **템포 평가** (`scripts/run_tempo_eval.py`): DSP, madmom, 1DSS 각각 CSV 3개 — `results/tempo_dsp.csv`, `tempo_madmom.csv`, `tempo_1dss.csv`. `1dss` 열은 jump-reward 가 필요해 **같은 스크립트를 `csc475-beat` 에서 돌려도 된다** (의존성 충돌 시).
- **비트 평가** (`scripts/run_beat_eval.py`): madmom, 1DSS — `results/beat_madmom.csv`, `beat_1dss.csv` (**GTZAN만**; GiantSteps는 비트 GT 없음).

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

평가 실행 예:

```bash
conda activate csc475-tempo
python scripts/run_tempo_eval.py --dataset both --sample-size 50

conda activate csc475-beat
python scripts/run_beat_eval.py --sample-size 50
```

- 오디오가 없으면 CSV `status=missing_audio`. 비트 평가는 **GTZAN만** (참 비트 파일이 있는 트랙).

4) 결과 확인
- 템포: `results/tempo_dsp.csv`, `tempo_madmom.csv`, `tempo_1dss.csv`
- 비트: `results/beat_madmom.csv`, `beat_1dss.csv`

1. For the final report the text will be copied and formatted as an ISMIR paper using LaTex. 

2. Code for possible uses: 
https://github.com/mjhydri/1D-StateSpace
madmom · PyPI
https://github.com/marsyas/marsgeyas


3. Papers to refer (pdfs are also in /papers folder):
Gouyon, Fabien, et al. "An experimental comparison of audio tempo induction algorithms." IEEE Transactions on Audio, Speech, and Language Processing 14.5 (2006): 1832-1844.

Percival, Graham, and George Tzanetakis. "Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses." IEEE/ACM Transactions on Audio, Speech, and Language Processing 22.12 (2014): 1765-1776.

Böck, Sebastian, and Markus Schedl. "Enhanced beat tracking with context-aware neural networks." Proc. Int. Conf. Digital Audio Effects. 2011.

4. datasets (샘플 진행 보고와 동일 계획)

- **GTZAN Tempo-Beat** — beat + global tempo, 다장르.
- **GiantSteps** — EDM·tempo 중심; annotation과 오디오 경로는 서브폴더마다 다를 수 있음.

로컬에는 `/dataset` 아래에 깃헙 repo를 통째로 두어도 된다. 권장: 역할별로 폴더명을 구분하고, 오디오가 없으면 해당 데이터셋 README의 **다운로드 스크립트**로만 받기. 대용량 오디오는 필요 시 `.gitignore`에 두고 경로만 문서화.

5. 비교 방법 (계획)

1. Autocorrelation / DSP baseline  
2. **madmom** (DBN beat tracking)  
3. **1D-StateSpace** — https://github.com/mjhydri/1D-StateSpace  

상세 일정·체크리스트는 `MIR_Project_Summary_Blueprint_Checklists.md` 참고.