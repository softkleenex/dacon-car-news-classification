# DACON 자동차 뉴스 분류 - GPT-4o-mini 프롬프트 엔지니어링

## 프로젝트 개요
**DACON 자동차 관련 뉴스 분류 경진대회**에서 GPT-4o-mini API를 활용한 프롬프트 엔지니어링 프로젝트입니다.

- **대회명**: 자동차 관련 뉴스 분류 경진대회
- **주최**: DACON
- **목표**: 뉴스 기사가 자동차 관련인지 이진 분류 (1: 자동차 관련, 0: 무관)
- **평가 모델**: GPT-4o-mini (temperature=0.4)
- **평가 지표**: `S = 0.9 × Accuracy + 0.1 × √(1-(bytes/3000)²)`

## 주요 성과

### 발견한 최적 프롬프트 TOP 5
| 순위 | 프롬프트 | 테스트 정확도 | 예상 점수 |
|------|----------|-------------|-----------|
| 1 | `자동차 아니면 0 맞으면 1` | 100% (46/46) | 0.910 |
| 2 | `주제를 분석한 후 답하세요. 자동차면 1, 아니면 0` | 100% (46/46) | 0.909 |
| 3 | `먼저 주제를 파악하고, 자동차 관련이면 1, 아니면 0만 출력` | 100% (46/46) | 0.908 |
| 4 | `차량 아니면 0, 맞으면 1.` | 93.5% (43/46) | 0.850 |
| 5 | `자동차 아니면 0, 맞으면 1.` | 91.3% (42/46) | 0.831 |

### 핵심 인사이트
- **구두점의 영향**: 마침표와 쉼표 제거 시 성능 향상 (91.3% → 100%)
- **언어별 성능**: 한국어가 영어보다 평균 6.5% 높은 정확도
- **Chain of Thought**: 단순 이진 분류에서는 오히려 성능 저하
- **길이 최적화**: 30바이트 이하가 최적 (점수 공식상 길이 페널티 최소화)

## 사용 방법

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/softkleenex/dacon-car-news-classification.git
cd dacon-car-news-classification

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정
```python
# config.py 파일 생성
cp config.py.example config.py

# config.py 편집
API_KEY = 'your-openai-api-key-here'
```

### 3. 테스트 실행
```bash
# 전체 프롬프트 테스트
python scripts/test_all_prompts.py

# 특정 프롬프트 테스트
python scripts/test_single_prompt.py "자동차 아니면 0 맞으면 1"

# Chain of Thought 테스트
python scripts/chain_of_thought_test.py

# 언어별 비교 테스트  
python scripts/english_vs_korean_test.py
```

## 📁 프로젝트 구조
```
dacon-car-news-classification/
│
├── README.md                    # 프로젝트 설명
├── requirements.txt             # 필요 패키지
├── config.py.example           # API 키 설정 템플릿
│
├── data/                       # 데이터 파일
│   ├── samples.csv            # 제공된 46개 샘플 데이터
│   └── synthetic_data.csv    # 생성한 합성 데이터 30개
│
├── scripts/                    # 실행 스크립트
│   ├── test_all_prompts.py   # 전체 프롬프트 테스트
│   ├── test_single_prompt.py # 단일 프롬프트 테스트
│   ├── chain_of_thought_test.py  # CoT 프롬프팅 테스트
│   ├── english_vs_korean_test.py # 언어별 비교
│   ├── synthetic_data_test.py    # 합성 데이터 테스트
│   └── evaluation_analysis.py    # 평가 공식 분석
│
├── docs/                       # 문서
│   ├── STRATEGY.md           # 전략 문서
│   ├── FINDINGS.md           # 실험 결과 정리
│   └── SUBMISSION_HISTORY.md # 제출 기록
│
└── results/                    # 테스트 결과
    ├── test_results.json      # 전체 테스트 결과
    └── submission_scores.csv # 실제 제출 점수
```

## 📈 평가 공식 분석

### 수학적 분석
```
S = 0.9 × Accuracy + 0.1 × √(1-(bytes/3000)²)

여기서:
- Accuracy: 정확도 (0~1)
- bytes: 프롬프트 UTF-8 인코딩 바이트 수
- 최대 길이: 3000바이트
```

### 점수별 필요 정확도
| 목표 점수 | 필요 정확도 (10바이트) | 필요 정확도 (100바이트) |
|-----------|------------------------|-------------------------|
| 0.95 | 94.4% | 94.5% |
| 0.90 | 88.9% | 88.9% |
| 0.85 | 83.3% | 83.4% |

## 🔬 실험 방법론

### 1. 기본 테스트
- 46개 샘플 전체에 대한 정확도 측정
- 자동차/비자동차 카테고리별 정확도 분석
- 프롬프트 길이에 따른 점수 계산

### 2. Chain of Thought (CoT)
- "Let's think step by step" 등 추론 강화 프롬프트
- 단계별 사고 프로세스 유도
- 이진 분류에서의 효과 검증

### 3. 언어별 비교
- 동일 의미의 한국어/영어 프롬프트 비교
- 한국어가 평균 6.5% 높은 성능 확인

### 4. 합성 데이터 검증
- 실제 뉴스 기반 30개 합성 데이터 생성
- 과적합 여부 검증
- 일반화 성능 평가

## 💡 주요 발견사항

### 성공 요인
1. **단순명료한 프롬프트**: 복잡한 조건보다 간단한 표현이 효과적
2. **구두점 제거**: 마침표, 쉼표 없는 프롬프트가 더 높은 정확도
3. **한국어 우선**: 한국 뉴스 데이터에는 한국어 프롬프트가 적합

### 실패 사례
1. **"자동차 1"**: 5/46 정확도 (10.9%) - 조건문 없이 실패
2. **특수문자 사용**: "자동차=1, 기타=0" - 파싱 오류 발생
3. **주관적 기준**: "핵심이면" 등 모호한 표현은 일관성 저하

## 📝 제출 기록
| 날짜 | 프롬프트 | 점수 | 비고 |
|------|----------|------|------|
| 2025-09-14 | `자동차 아니면 0, 맞으면 1.` | 0.856 | 최고 기록 |
| 2025-09-10 | `자동차 생산/판매/전기차부품이 핵심이면 1...` | 0.620 | 과적합 |

## 🛠️ 기술 스택
- **언어**: Python 3.x
- **API**: OpenAI GPT-4o-mini
- **라이브러리**: pandas, numpy, openai
- **분석 도구**: matplotlib, seaborn (시각화)

## 📚 참고 자료
- [DACON 대회 페이지](https://dacon.io/)
- [OpenAI API 문서](https://platform.openai.com/docs/)
- [프롬프트 엔지니어링 가이드](https://www.promptingguide.ai/)

## 🤝 기여
프로젝트 개선을 위한 Pull Request와 이슈 등록을 환영합니다!

## 📧 문의
- GitHub Issues: [프로젝트 이슈](https://github.com/softkleenex/dacon-car-news-classification/issues)

## 📄 라이선스
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Note**: 이 프로젝트는 DACON 경진대회 참가 경험을 정리한 것입니다.