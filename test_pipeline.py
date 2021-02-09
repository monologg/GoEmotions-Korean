from multilabel_pipeline import MultiLabelPipeline
from transformers import ElectraTokenizer
from model import ElectraForMultiLabelClassification
from pprint import pprint


tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-goemotions")
model = ElectraForMultiLabelClassification.from_pretrained("monologg/koelectra-base-v3-goemotions")

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