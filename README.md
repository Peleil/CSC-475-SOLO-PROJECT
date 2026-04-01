Beat Tracking and Tempo Estimation Project

## Quick Start (2주 크런치용)

1) 가상환경 생성/활성화 후 의존성 설치

```bash
pip install -r requirements.txt
```

2) GiantSteps만 먼저 쓸 때 (GTZAN 오디오는 나중에 받아도 됨)

- `dataset/giantsteps-tempo-dataset-master/README` 참고: 약 664개 mp3는 **별도 다운로드** (`audio_dl.sh` 또는 README의 Beatport URL 패턴).
- 받은 mp3를 아래처럼 두면 스크립트가 `1030011.LOFI.bpm` ↔ `1030011.LOFI.mp3` 로 매칭한다.

```text
dataset/giantsteps-tempo-dataset-master/audio/1030011.LOFI.mp3
```

```bash
python scripts/run_giantsteps_quick_check.py --sample-size 5
```

- 결과: `results/giantsteps_quick_check.csv` (`dsp_acc1` / `madmom_acc1` 은 상대 오차 4% 이내 여부)
- 오디오가 아직 없으면 `status=missing_audio` 로 남는다 → 정상, 다운로드 후 재실행.

3) (선택) 샘플 quick check — GiantSteps annotation만 있을 때

```bash
python scripts/run_quick_check.py --sample-size 3
```

4) 결과 확인
- `results/quick_check.csv` 또는 `results/giantsteps_quick_check.csv`
- `status`가 `missing_audio`이면 오디오 경로를 확인하거나 `--audio-root` 지정

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