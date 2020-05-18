# goemotions-korean

[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) 데이터셋을 한국어로 번역한 후, [KoELECTRA](https://github.com/monologg/KoELECTRA)로 학습

## Prepare translated data

- [pygoogletrans](https://github.com/ssut/py-googletrans)를 사용하여 한국어 데이터 생성
  - `v2.4.1`이 pypi에 업데이트되지 않은 관계로 repository에서 곧바로 라이브러리를 설치하는 것을 권장 (`requirements.txt`에 명시되어 있음)
- API 호출 간에 1.5초의 간격을 줌
- `​​&#x200B;`(Zero-width space)가 번역 문장 안에 있으면 번역이 되지 않는 오류가 있어서 이는 제거하였음

- 번역을 완료한 데이터는 `data` 디렉토리에 이미 있습니다. 혹여나 직접 번역하고 싶다면 아래의 코드를 실행하면 됩니다.

```bash
$ bash download_original_data.sh
$ pip3 install git+git://github.com/ssut/py-googletrans
$ python3 tranlate_data.py
```

## Train & Evaluation


## Results

## Pipeline

## Reference

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [googletrans](https://github.com/ssut/py-googletrans)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
