# README

The download links of the corpus for the paper are listed here.

## Download Links

|   Corpus   |   Link   |
| ---- | ---- |
|   **Train**   |   https://cloud.tsinghua.edu.cn/f/4c9caef008b546dbb0dc/   |
|   **Dev**   |   It will be available later.   |


## Data Description

One example is shown below:

```python
{
    "content": "世锦赛的整体水平远高于亚洲杯，要如同亚洲杯那样“鱼与熊掌兼得”，就需要各方面密切配合、#idiom#。作为主帅的俞觉敏，除了得打破保守思想，敢于破格用人，还得巧于用兵、#idiom#、灵活排阵，指挥得当，力争通过比赛推新人、出佳绩、出新的战斗力。", 
    "realCount": 2,
    "groundTruth": ["通力合作", "有的放矢"], 
    "candidates": [
        ["凭空捏造", "高头大马", "通力合作", "同舟共济", "和衷共济", "蓬头垢面", "紧锣密鼓"], 
        ["叫苦连天", "量体裁衣", "金榜题名", "百战不殆", "知彼知己", "有的放矢", "风流才子"]
    ]
}
```

- `content`: The given passage where the original idioms are replaced by placeholders `#idiom#`
- `realCount`: The number of placeholders or blanks
- `groundTruth`: The golden answers in the order of blanks
- `candidates`: The given candidates in the order of blanks
