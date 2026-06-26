# DACON 자동차 뉴스 분류 Prompt Engineering

DACON 자동차 관련 뉴스 분류 경진대회에서 GPT-4o-mini 평가 환경을 전제로, 짧은 프롬프트가 이진 분류 성능과 제출 점수에 어떤 영향을 주는지 실험한 프로젝트입니다.

## Problem

- **Task**: 뉴스 기사가 자동차 관련인지 분류
- **Output**: 자동차 관련 `1`, 무관 `0`
- **Evaluation model**: GPT-4o-mini, `temperature=0.4`
- **Score**: `0.9 * Accuracy + 0.1 * sqrt(1 - (bytes / 3000)^2)`
- **Constraint**: 모델 학습이나 코드 제출이 아니라 프롬프트만 제출

## Results

| 구분 | 값 |
| --- | --- |
| Best public submission | `0.856` |
| Best submitted prompt | `자동차 아니면 0, 맞으면 1.` |
| Best local validation prompt | `자동차 아니면 0 맞으면 1` |
| Local validation | `46/46` on provided samples |
| Extra robustness check | `86.7%` on 30 synthetic samples |

The main finding was counterintuitive but consistent across experiments: for this constrained binary task, a short Korean if/else instruction outperformed longer explanations and most Chain-of-Thought variants. Punctuation and wording changes also produced measurable differences.

## Key Findings

- **Simple Korean prompts worked best**: Korean prompts matched the Korean news context better than equivalent English prompts.
- **Length mattered less than accuracy, but still mattered**: the length term is only 10% of the score, so the prompt must stay short without losing the classification condition.
- **CoT was not a default win**: process-style prompts sometimes improved local accuracy, but extra bytes and output-format risk reduced their practical value.
- **Overfitting was real**: the best prompt reached `100%` on the 46 provided samples, then dropped on synthetic validation, so the final choice used both leaderboard history and local tests.

## Repository Structure

```text
.
├── README.md
├── config.py.example
├── data
│   ├── samples.csv
│   └── synthetic_data.csv
├── docs
│   ├── STRATEGY.md
│   └── SUBMISSION_HISTORY.md
├── requirements.txt
└── scripts
    ├── chain_of_thought_test.py
    ├── english_vs_korean_test.py
    ├── settings.py
    ├── test_all_prompts.py
    └── test_single_prompt.py
```

## Setup

```bash
git clone https://github.com/softkleenex/dacon-car-news-classification.git
cd dacon-car-news-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-openai-api-key"
```

`config.py.example` is kept for the older local-config workflow, but environment variables are preferred for a public repository.

Optional runtime settings:

```bash
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_TEMPERATURE="0.4"
export OPENAI_MAX_TOKENS="10"
export API_DELAY="0.02"
export FULL_TEST="true"
```

## Usage

```bash
# Test the full prompt list on the provided samples
python scripts/test_all_prompts.py

# Test one prompt with detailed error analysis
python scripts/test_single_prompt.py "자동차 아니면 0 맞으면 1"

# Compare Korean and English prompt variants
python scripts/english_vs_korean_test.py

# Compare standard prompts with CoT/process-style prompts
python scripts/chain_of_thought_test.py
```

## Experiment Notes

### Prompt candidates

| Prompt | Local accuracy | Estimated score |
| --- | ---: | ---: |
| `자동차 아니면 0 맞으면 1` | `100%` | `0.910` |
| `주제를 분석한 후 답하세요. 자동차면 1, 아니면 0` | `100%` | `0.909` |
| `먼저 주제를 파악하고, 자동차 관련이면 1, 아니면 0만 출력` | `100%` | `0.908` |
| `차량 아니면 0, 맞으면 1.` | `93.5%` | `0.850` |
| `자동차 아니면 0, 맞으면 1.` | `91.3%` | `0.831` |

### Failure patterns

- Missing conditions, such as `자동차 1`, caused severe failures.
- Subjective wording, such as "핵심이면", made the boundary unstable.
- English prompts underperformed on Korean news samples.
- Some prompts overfit the 46 provided examples and did not generalize as well to synthetic samples.

## Documentation

- [Strategy](docs/STRATEGY.md): problem analysis, prompt design, and risk management
- [Submission History](docs/SUBMISSION_HISTORY.md): actual submission scores and experiment tables

## License

MIT License. See [LICENSE](LICENSE).

<!-- BLOG-URL:START -->

## Blog

- Blog note: [DACON 자동차 뉴스 분류 Prompt Engineering](https://softkleenex.github.io/coding_training/dacon/dacon-car-news-classification)

<!-- BLOG-URL:END -->
