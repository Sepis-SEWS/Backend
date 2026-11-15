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

JSON에는 `encounter_id`, source `patient_id`, `bed`, `assignee`, `icu_id=ICU-A`, 시간(`admission_anchor` 또는 행별 `observed_at`)과 16개 특징이 필요합니다. ICULOS는 `1..n` 시간 격자입니다.

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
