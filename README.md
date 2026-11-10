# SEWS — ICU 패혈증 조기 경보 시스템

PhysioNet/CinC Challenge 2019의 ICU 환자 시간별 기록으로 **최근 24시간 관찰 → 패혈증 약 6시간 조기경보**를 구현하는 팀 프로젝트입니다. 과제/연구용 프로토타입으로, 실제 임상 적용이 검증된 의료기기가 아닙니다.

## 레포지토리 구성
- [AI](https://github.com/Sepis-SEWS/AI): 데이터·특징·XGBoost/LSTM·앙상블·검증·SHAP·Calibration
- [Backend](https://github.com/Sepis-SEWS/Backend): FastAPI 모델 서빙 및 예측 API
- [Frontend](https://github.com/Sepis-SEWS/Frontend): Streamlit 임상 모니터링 화면

## 공통 원칙
- 공식 `SepsisLabel`은 이미 발병 6시간 전에 양성으로 전환된 라벨이므로 **추가로 6시간 이동하지 않음**.
- 환자별 Train/Validation/Test 분리, 과거부터 현재까지 데이터만 사용, 평가 데이터로 전처리 통계 학습 금지.
- Streamlit은 AI 모델 파일을 직접 읽지 않음. 반드시 FastAPI를 호출.
- 성능 목표(미달성): AUROC ≥0.82, AUPRC ≥0.40, Brier ≤0.15, CPU 4코어 추론 평균 ≤200ms.
- 실험 결과 없이 달성 여부를 주장하지 않음. 실제 측정값과 예시를 구분.
- 환경: Python 3.9~3.11, PyTorch ≥2.0 또는 TensorFlow ≥2.10, Docker Compose, Ubuntu 22.04/WSL2 허용.
- 유형힌트·docstring·Conventional Commits·고정 라이브러리 버전 적용.

## API 계약 초안 (개발 전 협의 필요)
요청: 환자 식별자 + 현재 시점까지의 최근 24시간 기록(JSON). 응답: 위험 점수(`risk_score`), Calibration 점수(`calibrated_probability`), 위험 단계(`risk_level`), 계산 가능한 SOFA 요소와 SHAP 설명. 누락 필드는 null 또는 '계산 불가' 명시. `Warning 0.5~0.75`, `Alert ≥0.75`는 과제 예시이며 실제 임계값은 검증 데이터로 확정.

## 구현 주의
과제에 기재된 활력징후 8개/검사 10개 중 원본 칼럼명과 불일치하는 항목(RR·SpO2·INR·PaO2/FiO2)이 있음. 팀 계획의 원본 실제 변수는 활력징후 7개 + 검사 9개(총 16개). 누락 원천정보(GCS, 승압제, 소변량 등) 없이 완전한 SOFA/qSOFA 산출 불가. 대체 계산/미지원 상태를 명확히 기록해야 함.

## Backend 요구사항
FastAPI 사용, Pydantic 스키마 검사, API 문서 `/docs` 자동 제공.

| API | 역할 |
|---|---|
| `GET /health` | 서버 및 모델 로딩 상태 |
| `POST /predict` | 단일 환자 24시간 데이터 예측 |
| `POST /predict_batch` | 다수 환자 추론 |
| `POST /explain` | SHAP/LIME 주요 기여 요소 |

- 모델은 시작 시 한 번 로드(Singleton), 요청마다 재학습/재로드하지 않음.
- AI 레포에서 배포 승인된 모델 버전 + 특징 순서 + 전처리/보정 설정을 함께 사용.
- 입력 유효성/결측/시계열 정렬, 환자 ID와 기록 시간 정의를 AI 팀과 합의.
- API 응답에는 위험 점수, 위험 단계, 가능한 SOFA 점수, SHAP 설명 포함.
- CPU 4코어 참고 기준, batch=1/24h 입력, 10회 워밍업 + 100회 요청 평균 및 P95 지연 측정, 평균 ≤200ms 목표.
- Dockerfile 및 Compose 연동, `.env`로 민감정보 관리, 버전 고정.
- **현재 상태:** AI-10 최소 서버 코드 작성: `/health`, `/predict`, `/predict_batch`. 실제 모델 연동 실행 및 부하 검증은 로컬 확인 전. `/explain`과 Docker/Compose는 미구현.

## 구현 우선순위
입출력 JSON 계약 → `/health` → Mock `/predict` → AI 모델 실제 연동 → batch/explain → 지연/오류 검사.

## 팀 역할 및 개인 기여
추후 팀원별 담당/실제 커밋을 기록할 것.

## AI-10 로컬 FastAPI 연결 검증 (2026-11-10)

AI 저장소에서 승인된 단일 **미보정** LSTM을 로딩하는 최소 서버입니다. 모델은 시작할 때 **1회** 로딩하고, 다른 폴드 체크포인트나 새로운 모델로 바꾸지 않습니다. 원래 잠금 정책의 threshold=0.2, 최초 23개 기록 무경보를 그대로 유지합니다.

**입력 계약 정정:** 위 초기 API 초안의 '최근 24시간 기록'만으로는 학습 입력을 재현할 수 없습니다. 학습 전처리에는 각 변수의 마지막 실측 시점 이후 경과시간 채널이 있으므로, **현재 시점까지 누적된 환자별 전체 시간 기록**을 입력해야 합니다. 현재 API는 환자별 전체 기록을 각 요청에 전달하는 형태이며, 서버가 환자 상태를 저장하지 않습니다.

데스크탑 Windows PowerShell 실행 예 (AI 저장소는 `C:\AI`에 있다고 가정):

```powershell
cd C:\
git clone https://github.com/Sepis-SEWS/Backend.git
cd C:\Backend
python -m pip install -r requirements.txt
# AI 저장소에도 torch/pandas/numpy 등 학습·추론 의존성이 설치돼 있어야 합니다.
$env:SEWS_AI_REPO_ROOT = "C:\AI"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

다른 터미널에서:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
# 요청 스키마는 http://127.0.0.1:8000/docs 에서 확인
```

테스트 전용 의존성 설치 후, 실제 모델을 로딩하지 않는 **합성 데이터 API 계약 테스트**:

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

**요청 예시 (축약; 실제 운영에서는 16개 원본 변수를 각 시점에 모두 명시)**:

```json
{
  "patient_id": "synthetic-001",
  "observations": [
    {"ICULOS": 1, "HR": 80.0, "Temp": null}
  ]
}
```

이 예시의 관측값은 **실제 요청에 필요한 전체 변수 집합을 생략한 스키마 설명용**이므로, 모델 예측에 그대로 사용하지 않습니다. 누락 변수는 AI 공통 전처리 단계에서 422로 거부됩니다. 모든 변수는 관측되지 않은 경우 null로 제공하세요.

`POST /predict`는 환자 ID와 `status`, `risk_score`, `alert`, `score_threshold`, `records_received`, `minimum_records`, `calibrated_probability=null`을 반환합니다. 24행 미만은 `risk_score=null`, `alert=false`; 모델 점수는 **임상적 발생 확률이 아닙니다.** `POST /predict_batch`는 최대 32개 환자를 순차적으로 추론하며 중복 patient_id를 거부합니다. 로그와 API 요청에는 실제 환자식별정보를 사용하지 마세요.

미구현 항목: `/explain`, SOFA/qSOFA 전체 점수, 확률 보정 모델, 임상 위험 단계별 별도 임계값, 실제 HTTP 평균/P95 지연 측정, 배포용 인증·접근제어·Docker/Compose. **최종 Test 데이터는 다시 평가하지 않습니다.**
