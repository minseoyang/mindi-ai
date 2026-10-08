# MINDI

### 음성·텍스트 분석과 회상 대화를 결합한 인지 케어 플랫폼

어르신이 음성으로 질문에 답하고 대화에 참여하면, 음향·언어 특징을 분석하고 결과 리포트를 제공하는 프로젝트입니다. 분석 기능과 일상·회상 대화 기능을 함께 구성했습니다.

2025년 세종대학교 캡스톤디자인에서 개발한 **학습·연구용 프로토타입**입니다. 모델 점수는 프로젝트 내부의 실험 지표입니다.

| 항목 | 내용 |
| --- | --- |
| 개발 시기 | 2025년 1학기 |
| 개발 형태 | 3인 팀 프로젝트 |
| 담당 역할 | AI 개발, 모델 결합 및 파이프라인 통합 |
| 주요 기술 | Python, PyTorch, Transformers, Whisper, BERT/KoBERT, OpenAI API |

## TASK
### 개발

- **음성 전사·전처리**: Whisper 기반 STT, 음성 응답의 텍스트 변환, 발화 텍스트 정제
- **텍스트 분석**: BERT/KoBERT 기반 분류, GPT 기반 질문·응답 의미 평가와 대화 문맥 연결성 평가
- **회상 대화**: 일상·회상 질문 프롬프트, 주제 선택, 대화 로그 저장과 핵심 키워드 추출
- **모델 결합·리포트**: 문항·텍스트·음향 분석 결과를 연결하는 점수 흐름, 대화 평가와 주간 요약 리포트

### 협업·통합

음향 모델 담당 팀원과 설계 방향 및 개선점을 검토하고 피드백을 제공했습니다. 음향 모델 결과를 텍스트 모델·문항 평가와 연결하는 전체 AI 파이프라인 통합을 주도했습니다.

## 주요 기능

| 기능 | 구현 내용 |
| --- | --- |
| 음성 전사 | 음성을 텍스트로 변환하여 모델 입력으로 사용 |
| 문항 응답 평가 | 질문·기준 답변·사용자 응답을 GPT에 전달하여 의미 일치 여부 평가 |
| 언어 특징 분석 | BERT/KoBERT 분류 결과와 질문·응답의 문맥 연결성을 함께 활용 |
| 음향 특징 분석 | Mel 스펙트로그램과 음향 수치 특징을 분석하는 모델 결과 결합 |
| 일상·회상 대화 | 이전 대화 기록을 활용한 회상 질문과 주제별 일상 질문 생성 |
| 기록·키워드 추출 | 날짜별 대화 기록 저장과 핵심 키워드 추출 모듈 |
| 리포트 | 회상 충실도, 언어 유창성, 맥락 일관성, 정서 반응, 문장 구성력 평가 및 주간 요약 |

## AI 처리 구조

음성 입력에서 모델 분석과 결과 리포트까지의 흐름입니다. **음향·언어 분석**과 **일상·회상 대화 케어**를 구분해 표시했습니다.

![MINDI AI 처리 구조: 음성 응답은 음향 분석과 Whisper 전사로 나뉘고, 전사 텍스트는 BERT·KoBERT 분류 및 GPT 평가에 사용됩니다. 분석 결과는 결합되어 리포트로 정리되며, 별도 대화 케어 흐름은 대화 생성, 기록·키워드 추출, 평가·주간 요약으로 이어집니다.](docs/assets/mindi-ai-pipeline.svg)

## 기술과 모듈

| 영역 | 기술 / 접근 방식 |
| --- | --- |
| 음성 전사 | Whisper, PyTorch, Transformers, librosa, torchaudio |
| 텍스트 분석 | BERT/KoBERT, 문장 분류, GPT 기반 의미·문맥 평가 |
| 음향 분석 | ViT, Mel 스펙트로그램, MFCC, pitch, jitter, shimmer, HNR |
| 영어 음향 모델 | ViT와 LightGBM 결과 결합 |
| 한국어 음향 확장 | ViT와 PyTorch 기반 음향 특징 분류기 조합 |
| 데이터 처리 | pandas, NumPy, scikit-learn |
| 대화·리포트 | OpenAI API, 날짜별 대화 기록, 프롬프트 기반 평가·요약 |

## 코드 구성

| 위치 | 내용 |
| --- | --- |
| [model.py](model.py) | 영어 분석 모델 결합 실험 |
| [ko_model.py](ko_model.py) | 한국어 분석 모델 결합 실험 |
| [data_make_csv.py](data_make_csv.py), [ko_data_make_csv.py](ko_data_make_csv.py), [wav_text.py](wav_text.py) | 음성 전사 및 전사 데이터 구성 |
| [addresso_data_cha_to_csv.py](addresso_data_cha_to_csv.py) | 전사 파일에서 대상 발화 추출·정제 |
| [language_model](language_model/) | 영어 BERT 추론과 GPT 응답 평가 |
| [ko_language_model](ko_language_model/) | 한국어 KoBERT 추론과 GPT 응답 평가 |
| [acoustic_model](acoustic_model/) | 영어 ViT·LightGBM 음향 분석 |
| [ko_acoustic_model](ko_acoustic_model/) | 한국어 음향 확장 실험 |
| [care_model](care_model/) | 일상·회상 대화, 기록, 키워드 추출 |
| [report](report/) | 대화 평가와 리포트 생성 |
