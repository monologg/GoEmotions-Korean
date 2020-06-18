# GoEmotions-Korean

[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) 데이터셋을 한국어로 번역한 후, [KoELECTRA](https://github.com/monologg/KoELECTRA)로 학습

## Updates

**June 19, 2020** - Transformers v2.9.1 기준으로 모델 학습 시 `[NAME]`, `[RELIGION]`과 같은 Special token을 추가하였음에도 pipeline에서 다시 사용할 때 적용이 되지 않는 이슈가 있었으나, Transformers v2.11.0에서 해당 이슈가 해결되었습니다.

## GoEmotions

**58000개의 Reddit comments**를 **28개의 emotion**으로 라벨링한 데이터셋

- admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

## Requirements

- torch==1.4.0
- transformers=2.11.0
- googletrans==2.4.1
- attrdict==2.0.1

```bash
$ pip3 install -r requirements.txt
```

## Translated Data

🚨 **Reddit 댓글로 만든 데이터여서 번역된 결과물의 품질이 좋지 않습니다.** 🚨

- [pygoogletrans](https://github.com/ssut/py-googletrans)를 사용하여 한국어 데이터 생성
  - `pygoogletrans v2.4.1`이 pypi에 업데이트되지 않은 관계로 repository에서 곧바로 라이브러리를 설치하는 것을 권장 (`requirements.txt`에 명시되어 있음)
- API 호출 간에 1.5초의 간격을 주었습니다.
  - 한 번의 request에 최대 5000자를 넣을 수 있는 점을 고려하여 문장들을 `\r\n`으로 이어 붙여 input으로 넣었습니다.
- `​​&#x200B;`(Zero-width space)가 번역 문장 안에 있으면 번역이 되지 않는 오류가 있어서 이는 제거하였습니다.
- **번역을 완료한 데이터는 `data` 디렉토리에 이미 있습니다.** 혹여나 직접 번역을 돌리고 싶다면 아래의 명령어를 실행하면 됩니다.

```bash
$ bash download_original_data.sh
$ pip3 install git+git://github.com/ssut/py-googletrans
$ python3 tranlate_data.py
```

## Tokenizer

- 데이터셋에 `[NAME]`, `[RELIGION]`의 Special Token이 존재하여, 이를 `vocab.txt`의 `[unused0]`와 `[unused1]`에 각각 할당하였습니다.
- `transformers v2.9.1` 기준으로 `additional_special_tokens`에 위의 두 개의 토큰을 추가하였음에도 처리가 되지 않는 이슈가 있어 **config를 통해서가 아닌 code 단에서 직접 넣어줘야 합니다**. (Pipeline 코드 참고)

## Train & Evaluation

- Sigmoid를 적용한 Multi-label classification (**threshold는 0.3으로 지정**)
  - `model.py`의 `ElectraForMultiLabelClassification` 참고
- config의 경우 `config` 디렉토리의 json 파일에서 변경하면 됩니다.

```bash
$ python3 run_goemotions.py --config_file koelectra-base.json
$ python3 run_goemotions.py --config_file koelectra-small.json
```

## Results

`Macro F1`을 기준으로 결과 측정 (Best result)

| Macro F1 (%)        |  Dev  |   Test    |
| ------------------- | :---: | :-------: |
| **KoELECTRA-Small** | 36.92 | **37.87** |
| **KoELECTRA-Base**  | 40.34 | **41.54** |

## Pipeline

- `MultiLabelPipeline` 클래스를 새로 만들어 Multi-label classification에 대한 inference가 가능하게 하였습니다.
- Huggingface s3에 `monologg/koelectra-base-finetuned-goemotions`와 `monologg/koelectra-small-finetuned-goemotions` 모델을 업로드하였습니다.

```python
from multilabel_pipeline import MultiLabelPipeline
from transformers import ElectraTokenizer
from model import ElectraForMultiLabelClassification
from pprint import pprint


tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-finetuned-goemotions")
model = ElectraForMultiLabelClassification.from_pretrained("monologg/koelectra-base-finetuned-goemotions")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)

texts = [
    "전혀 재미 있지 않습니다 ...",
    "나는 “지금 가장 큰 두려움은 내 상자 안에 사는 것” 이라고 말했다.",
    "곱창... 한시간반 기다릴 맛은 아님!",
    "애정하는 공간을 애정하는 사람들로 채울때",
    "너무 좋아",
    "딥러닝을 짝사랑중인 학생입니다!",
    "마음이 급해진다.",
    "아니 진짜 다들 미쳤나봨ㅋㅋㅋ",
    "개노잼"
]

pprint(goemotions(texts))

# Output
[{'labels': ['disapproval'], 'scores': [0.8736532]},
 {'labels': ['fear'], 'scores': [0.97049415]},
 {'labels': ['neutral'], 'scores': [0.94366586]},
 {'labels': ['neutral'], 'scores': [0.49382192]},
 {'labels': ['admiration'], 'scores': [0.9761528]},
 {'labels': ['love', 'neutral'], 'scores': [0.4782611, 0.5644545]},
 {'labels': ['caring', 'nervousness'], 'scores': [0.31030405, 0.76883996]},
 {'labels': ['amusement'], 'scores': [0.9888398]},
 {'labels': ['annoyance'], 'scores': [0.93368]}]
```

## Reference

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [googletrans](https://github.com/ssut/py-googletrans)
