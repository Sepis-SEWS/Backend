# SEWS Backend 설계 요약

**목적:** ICU 관측 데이터 → 패혈증 위험 점수 계산 → 경보·추이 표시.
**범위:** 비식별 데이터 기반 로컬 연구용. 실제 진료용 아님.

## 데이터 흐름

```text
비식별 JSON → 입력 검증 → AI LSTM 추론 → SQLite 저장
                                           ↕
                                  FastAPI ↔ Streamlit
```

- **입력:** 입원별 ID·병상·담당자·시각 + 시간당 관측값.
- **활력징후 7개:** HR, Temp, SBP, DBP, MAP, Resp, O2Sat.
- **검사 9개:** WBC, Lactate, Creatinine, Bilirubin_total, Platelets, FiO2, BUN, Glucose, pH.
- **결측:** `null` 유지. 라벨·잘못된 시각·무한값 거부.
- **연구 데이터:** PhysioNet 2019 전체를 변환해 적재(3-1절). 시각은 원본 관측 시각이 아닌 재생 시각.
- **추론:** 전체 누적 이력 전달 → 최근 24행 모델 입력 구성. 미래 데이터 사용 금지.

## 위험도 점수

모델 출력 **0–1** → 화면 표시 **0–100점** (`모델 점수 × 100`).

| 구분 | 모델 출력 | 화면 점수 | 경보 |
|---|---|---|---|
| 낮음 `low` | 0 이상 ~ 0.2 미만 | 0 이상 ~ 20점 미만 | 없음 |
| 높음 `high` | 0.2 이상 ~ 1 이하 | 20점 이상 ~ 100점 이하 | 생성 |
| 계산 불가 `unavailable` | 관측 24행 미만 → `null` | 미표시 | 없음 |

- **기준:** 기존 승인 모델의 고정 임계값 `0.2`. 20점부터 경보.
- **주의 `warning`:** 별도 검증 임계값 없음 → 현재 미사용.
- **의미:** 모델 점수이지 발생 확률 아님. **20점 ≠ 패혈증 확률 20%.**
- **경보:** 양성인 매시간 독립 생성. 동일 파일 재입력으로 중복 생성하지 않음.

## 역할 분리

| 구성 | 역할 |
|---|---|
| AI | 고정 LSTM·학습 전처리 통계·정책 검증, 프로세스당 1회 로드 |
| 파일 가져오기 CLI | 관측 검증·시간순 추론·결과 저장 |
| SQLite | 관측·예측·경보·확인·조치기록 보관 |
| FastAPI | 환자·경보 조회, 확인·기록 처리 |
| Streamlit | 저장 결과 조회·점수·추이 표시. 화면 갱신 시 재추론 없음 |

- **순수 추론:** `/predict`, `/predict_batch` → DB 저장 없음.
- **대시보드:** `/patients`, `/patients/{id}`, `/alerts`.
- **확인·기록:** 경보 확인 후 조치 저장. 반복 요청 중복 방지.
- **상태:** `/health` → 모델 준비·DB 상태 분리 확인.

## 실행에 필요한 파일

세 저장소가 같은 부모 폴더에 있어야 합니다.

```text
Sepsis-SEWS.Team/
├── AI/
│   ├── models/
│   │   ├── lstm_full_pw10_best.pt          # GitHub에 없음. 별도 전달
│   │   └── lstm_preprocessing_stats.json   # GitHub에 없음. 별도 전달
│   ├── src/inference/predict_lstm.py
│   └── reports/final_evaluation_policy_locked.json
├── Backend/
└── Frontend/
```

| 파일 | 역할 |
|---|---|
| `AI/models/lstm_full_pw10_best.pt` | 승인 LSTM 가중치 |
| `AI/models/lstm_preprocessing_stats.json` | 학습 당시 전처리 통계 |
| `AI/reports/final_evaluation_policy_locked.json` | 체크섬·임계값 `0.2` 검증 |
| `AI/src/inference/predict_lstm.py` | 유일한 추론 구현 |
| `Backend/requirements.txt` | FastAPI 실행 의존성 |
| `AI/requirements.txt` | torch 등 모델 로드 의존성 |
| `tools/physionet2019/` | PhysioNet 2019 변환·검증·일괄 실행(`run`) 스크립트 |
| `AI/data/data-sepsis/training/` | PhysioNet 2019 원본(팀 공통 위치, 커밋 안 함) |
| `Frontend/requirements.txt` | Streamlit 실행 의존성 |
| 비식별 JSON | 대시보드에 넣을 관측 데이터. `/predict`만으로는 저장되지 않음 |

모델 두 파일은 `.gitignore` 대상이라 `git pull`로 받지 않습니다. `.pt`는 압축을 풀지 말고 파일 그대로 `AI/models/`에 둡니다. 체크섬이 정책 파일과 같아야 로드됩니다.

설정 예시는 [.env.example](.env.example)입니다. 서버가 `.env`를 자동으로 읽지 않으므로 아래처럼 환경변수를 직접 설정합니다.

## 실행

Python **3.11**. 터미널 두 개를 사용합니다. **백엔드를 먼저** 켭니다.

### 1. 설치 (최초 1회)

Backend:

```sh
cd Backend
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r ../AI/requirements.txt
```

Frontend:

```sh
cd Frontend
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2. 백엔드

```sh
cd Backend
export SEWS_AI_REPO_ROOT="$(pwd)/../AI"
export SEWS_DATABASE_PATH="$(pwd)/data/sews.sqlite"
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

확인: http://127.0.0.1:8000/health  
`status=ok`, `model_ready=true`이면 실제 모델이 로드된 상태입니다. 모델 파일이 없으면 `degraded`이며 가짜 점수로 대체하지 않습니다.

Windows PowerShell:

```powershell
cd Backend
$env:SEWS_AI_REPO_ROOT = (Resolve-Path ..\AI).Path
$env:SEWS_DATABASE_PATH = (Join-Path (Get-Location) 'data\sews.sqlite')
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

### 3. 관측 가져오기 (대시보드에 환자를 넣을 때)

백엔드와 **같은** `SEWS_DATABASE_PATH`를 사용합니다.

```sh
cd Backend
export SEWS_AI_REPO_ROOT="$(pwd)/../AI"
export SEWS_DATABASE_PATH="$(pwd)/data/sews.sqlite"
.venv/bin/python import_cli.py 관측파일.json
```

JSON에는 `encounter_id`, source `patient_id`, `bed`, `assignee`, `icu_id=ICU-A`, 시간(`admission_anchor` 또는 행별 `observed_at`)과 16개 특징이 필요합니다. ICULOS는 원본 값 그대로이며, 첫 값이 1 이상의 정수이고 이후 1시간씩 빠짐없이 이어져야 합니다(예: 1,2,3… 또는 5,6,7…).

### 3-1. PhysioNet 2019 전체 데이터로 로컬 DB 만들기

원본과 변환 결과는 **커밋하지 않습니다.** 팀 공통 위치(`AI/data/data-sepsis/training`)의 원본으로 같은 스크립트를 돌리면 같은 변환 JSON(바이트 단위)과 같은 환자·관측값·점수·경보 시각의 DB가 만들어집니다. 예측·경보 ID와 계산 시각만 다릅니다. `data/`는 `.gitignore` 대상입니다.

**한 번에 실행 (권장)**

Backend 폴더에서 명령 하나로 변환 → 검증 → 적재 → 결과 요약까지 진행합니다.

```sh
cd Backend
.venv/bin/python -m tools.physionet2019.run
```

| 단계 | 하는 일 | 기록 |
|---|---|---|
| 1. 변환 | 원본 40,336명 전부 `.psv` → JSON. 라벨 제외, 결측 `null`, 단위 유지, 재생 시각 부여 | `data/import/physionet2019/_conversion_report.json` (데이터 버전·원본 해시·대상 목록·실패 원인) |
| 2. 검증 | 백엔드 규격·라벨 부재·ICULOS 격자·재생 시각 표시·원본 값과 일치 | 화면 `valid N/N`, 실패 시 `FAIL` 줄 |
| 3. 적재 | `import_cli.py`로 DB 적재. 1%마다 진행률 표시 | `data/import/_import_report.json` (성공·실패 건수, 실패 원인), `data/import/_import_log.jsonl` |
| 4. 요약 | `/patients`, `/patients/{id}`, `/alerts`에 보일 내용 | 화면 |

경로는 자동으로 정해집니다. 다르면 옵션으로 바꿉니다.

| 항목 | 우선순위 |
|---|---|
| 원본 | `--source-root` → `$SEWS_AI_REPO_ROOT/data/data-sepsis/training` → `../AI/data/data-sepsis/training` |
| 모델 | `--ai-root` → `$SEWS_AI_REPO_ROOT` → `../AI` |
| DB | `--database` → `$SEWS_DATABASE_PATH` → `Backend/data/sews.sqlite` |

- `--limit 50`: 세트별 50명만 빠르게 시험합니다.
- `--skip-convert`: 이미 변환한 JSON을 재사용합니다.
- `--skip-validate`: 검증 단계를 건너뜁니다.
- 검증에 실패하거나 모델 파일이 없으면 적재 전에 이유를 출력하고 멈춥니다(종료 코드 1).
- 적재가 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 진행합니다.
- 실행이 끝나면 화면 마지막에 출력되는 `SEWS_DATABASE_PATH`로 API 서버를 띄웁니다.

정상 실행 시 화면 예시입니다(가짜 모델 기준이라 경보·위험도 수치는 실제 모델과 다릅니다).

```text
[1/4] convert .psv -> import JSON
  converted 40336/40336, failed 0
[2/4] validate converted JSON against contract and originals
valid 40336/40336
[3/4] import into the local DB (resumable; log in _import_log.jsonl)
  files 40336/40336 imported, 0 failed ; rows added 1552210, alerts added …
[4/4] what the API will show
  /patients        total 40336  latest risk {'high': …, 'low': …, 'unavailable': 9675}
  /patients/{id}   e.g. /patients/pn19-p000001  (replay-time marked: 40336/40336)
  /alerts          total …
```

아래 1)–4)는 각 단계를 따로 실행하거나 문제를 확인할 때 참고합니다.

**1) 원본 위치 (커밋 안 함)**

원본은 팀 공통 위치인 AI 저장소의 `data/data-sepsis/training/`에 둡니다. 스크립트가 이 경로를 자동으로 찾습니다.

```text
DMUsews/
├── AI/data/data-sepsis/training/
│   ├── training_setA/p000001.psv …   (20,336개)
│   └── training_setB/p100001.psv …   (20,000개)
└── Backend/
```

- 경로를 정하는 순서: `--source-root` 옵션 → `$SEWS_AI_REPO_ROOT/data/data-sepsis/training` → `Backend/../AI/data/data-sepsis/training`.
- 현재 폴더와 상관없이 Backend 저장소 위치를 기준으로 계산합니다.
- 원본이 다른 곳에 있으면 `--source-root 경로`로 지정합니다.

| 항목 | 값 |
|---|---|
| 데이터 | PhysioNet/CinC Challenge 2019 **v1.0.0** 학습셋 전체 |
| 대상 | `training_setA` 20,336명 + `training_setB` 20,000명 = **40,336명 전부** (선별 없음) |
| 공식 경로 | https://physionet.org/content/challenge-2019/1.0.0/ (`training/training_setA`, `training/training_setB`) |
| 팀 공유본 | `아카이브.zip` SHA-256 `b54e55854e6c9aa75a21c1947f970e5162bd1b0ad14957279bac1ab60277bb57` |
| 원본 동일성 기준 | 변환 보고서의 `source_listing_sha256` = `96cb3e1100f7cd90484655096d1c4dd403b8f65f552f0b6bb66651d9d8dec79b` |

원본이 아직 없다면 팀 공유본을 그 위치에 풉니다(`__MACOSX`는 제외).

```sh
unzip 아카이브.zip -x '__MACOSX/*' -d ../AI/data/data-sepsis/training
```

**2) 변환 규칙**

- 입력 특징: `ICULOS` + 16개(HR, Temp, SBP, DBP, MAP, Resp, O2Sat, WBC, Lactate, Creatinine, Bilirubin_total, Platelets, FiO2, BUN, Glucose, pH). 나머지 열과 **`SepsisLabel`은 읽지도 쓰지도 않습니다.**
- 단위: `physionet2019-source-v1` = PhysioNet 원본 단위(모델 전처리 통계도 원본 값 기준). **단위 변환·클리핑·보간 없음.** FiO2는 원본처럼 0–1 비율 값입니다. 범위를 벗어난 원본 25건도 그대로 두고 보고서 `warnings`에만 표시합니다.
- 결측 `NaN` → `null`.
- ICULOS: 원본 값 그대로입니다. 8,793명은 첫 ICULOS가 1이 아니라 2 이상(최대 304)에서 시작하는데, 행을 채우거나 당기지 않고 그대로 씁니다. 백엔드는 "첫 값이 1 이상의 정수이고 이후 1시간씩 빠짐없이 이어지는" 기록을 받습니다(예: 1,2,3… 또는 5,6,7…).
- 시간: 원본에 실제 날짜가 없으므로 **연구 재생 시각**을 부여합니다. `admission_anchor = 2026-01-01T00:00:00Z`이고, 각 행은 `anchor + ICULOS시간`입니다. 파일의 `provenance.time_basis = "research_replay"`가 원본 관측 시각이 아님을 명시하며, DB `encounters.envelope`에도 저장됩니다. API는 항상 `data_delayed=true`, `delay_reason="replay_historical"`로 표시합니다.
- ID: `encounter_id = pn19-<기록ID>`, `patient_id = <기록ID>`, 병상 `BED-<기록ID>`, 담당자 `research-rn-1..4`(가상, 기록 번호 기준 고정).
- 재생 기준 시각을 `--replay-base`로 바꾸면 DB 해시가 달라져 같은 DB에 다시 넣을 때 `import_conflict`가 납니다. 팀 기준에서는 바꾸지 마세요.

**3) 변환 → 검증 → 적재**

```sh
cd Backend
export SEWS_AI_REPO_ROOT="$(pwd)/../AI"
export SEWS_DATABASE_PATH="$(pwd)/data/sews.sqlite"

# 변환: ../AI/data/data-sepsis/training → data/import/physionet2019/pn19-*.json (40,336개, 약 460MB, 1–2분)
.venv/bin/python -m tools.physionet2019.convert
#   빠른 확인용: --limit 50  (세트별 앞 50명만)

# 검증(선택, DB 미사용): 스키마·라벨 부재·ICULOS 격자·재생 시각·원본 값/해시 일치 (--no-source: 원본 비교 생략)
.venv/bin/python -m tools.physionet2019.validate

# 적재: 폴더를 주면 *.json(밑줄로 시작하는 보고서 제외)을 모두 처리
.venv/bin/python import_cli.py data/import/physionet2019 --report data/import/_import_report.json
```

Windows PowerShell은 `.venv\Scripts\python.exe`를 쓰고, 환경변수는 2절과 같이 설정합니다.

적재는 환자마다 매 시간 모델을 호출하므로(총 약 155만 회) 실제 LSTM으로는 오래 걸립니다. 파일 단위로 커밋되므로 **중간에 끊겨도 같은 명령을 다시 실행하면** 이미 들어간 시간은 건너뛰고 이어서 적재합니다.

**4) 결과 확인**

- 변환: 마지막 줄의 `conversion_summary`가 `converted 40336, failed 0`이어야 합니다. `source_listing_sha256`이 위 표의 값과 같아야 같은 원본입니다. 자세한 내용은 `data/import/physionet2019/_conversion_report.json`에 있습니다(데이터 버전, 레코드별 sha256·행 수·ICULOS 시작값·실패 코드).
- 적재 출력:
  - stdout: 파일마다 성공 요약 한 줄.
  - stderr: 실패는 `{"error":{"code",…},"file_index":n}`.
  - 마지막 줄: `{"import_summary":{"files_total","files_imported","files_failed","records_added","alerts_added","failure_codes"}}`.
  - 실패가 하나라도 있으면 종료 코드 1이지만, **나머지 파일은 계속 적재**합니다.
- `--report` 파일에는 파일 경로별 결과와 실패 코드·메시지, 스키마 오류 위치(`loc`/`type`/`msg`)가 들어갑니다. 입력값은 기록하지 않습니다. 콘솔 출력에는 파일명·내용을 노출하지 않습니다.
- 주요 실패 코드:
  - `invalid_import`: 규격 불일치
  - `file_unreadable`, `invalid_encoding`: 파일을 읽을 수 없음
  - `import_conflict`: 같은 encounter_id에 다른 내용
  - `future_observations`: 재생 시각이 현재보다 미래
  - `inference_failed`: 모델 추론 실패
- 다시 실행하면 `records_added=0`입니다(중복 적재 없음).
- API는 서버를 같은 `SEWS_DATABASE_PATH`로 띄운 뒤 확인합니다.
  - `GET /patients`: total 40,336, 그중 24행 미만 9,675명은 `unavailable`.
  - `GET /patients/pn19-p000001`
  - `GET /alerts?patient_id=pn19-p000001`
  - 전체 목록 응답은 수 초 이상 걸릴 수 있습니다.

DB를 처음부터 다시 만들려면 서버를 끄고 `data/sews.sqlite*`를 지운 뒤 적재를 다시 실행합니다.

### 4. 프론트 API 모드

**다른 터미널.** 백엔드가 떠 있는 상태에서:

```sh
cd Frontend
export SEWS_MODE=api
export API_BASE_URL=http://127.0.0.1:8000
.venv/bin/streamlit run app.py
```

화면: http://127.0.0.1:8501  
헤더에 **연결 정상**이 보이면 백엔드·실제 모델과 붙은 상태입니다. 데모 점수로 바뀌지 않습니다.

백엔드가 토큰을 요구할 때만, Backend와 Frontend에 같은 값을 넣습니다.

```sh
export SEWS_API_TOKEN='로컬-개발용-토큰'
```

### 5. 프론트 데모만 (백엔드 없이)

```sh
cd Frontend
.venv/bin/streamlit run app.py
```

합성 데이터입니다. 실제 모델 결과가 아닙니다.

종료는 각 터미널에서 `Ctrl+C`입니다.

## 운영 기준·제한

- FastAPI worker 1개, `--host 127.0.0.1`. access log는 끄세요.
- API와 CLI는 같은 DB 절대 경로를 써야 합니다.
- 입력은 시간당 비식별 파일 가져오기입니다. 실시간 병원 수집이 아닙니다.
- 모델 미준비 시 추론 503. 가짜 점수 대체 없음.
- 미지원: 보정 확률·SHAP·완전 SOFA·임상 운영 인증.
