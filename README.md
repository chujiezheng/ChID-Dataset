# ChID-Dataset

The ChID Dataset for paper **[ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://www.aclweb.org/anthology/P19-1075)**.

If you have any problem or suggestion, feel free to contact me: chujiezhengchn@gmail.com

If your research is related to or based on our ChID dataset (or the version adapted for the competition), please kindly cite it:

```bib
@inproceedings{zheng-etal-2019-chid,
    title = "{C}h{ID}: A Large-scale {C}hinese {ID}iom Dataset for Cloze Test",
    author = "Zheng, Chujie  and
      Huang, Minlie  and
      Sun, Aixin",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1075",
    pages = "778--787",
}
```

## Download Link

[Here](https://drive.google.com/drive/folders/1qdcMgCuK9d93vLVYJRvaSLunHUsGf50u?usp=sharing).


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

## Baseline Codes

Please refer to `Codes for baseline`.

## Competition

We are organizing a [competition](https://biendata.com/competition/idiom/) adapted from the ChID dataset. For the adapted data and baseline codes of the competition, please refer to `Competition`.


## Update History

### Update 191001

The competition has finished. We have uploaded **all split sets** of ChID! Feel free to use it in your research.

### Update 190702

The file `wordList.txt` used in baselines (both for paper and for competition) has been uploaded. 

Note that due to the potential differences in equipments and word segmentation tools, your segmentation results may not perfectly match with the vocabulary we provide. For the sake of performance, we suggest you do the segmentation and get the vocabulary list by yourself.

### Competition

We are organizing a competition based on our ChID dataset, and [here](https://biendata.com/competition/idiom/) is the website. The adapted corpus establishes up connections between blanks, and adopts a new type of problem. A list of passages (not an isolated one) is provided and the answers need to be selected from a given set of candidate idioms with fixed length (for more details, please refer to the competition website). 

The public data contains the training data (both the passages with blanks and the golden answers), the development data (only the passages, and the answers will be available later) and the corpus of idiom explanations.

The baselines of the competition are [Attentive Reader](https://arxiv.org/abs/1506.03340) and [BERT](https://arxiv.org/abs/1810.04805).

The data and baseline codes for the competition will be uploaded later.

### Update 190605

The download link of **Train** corpus is available! Please refer to `Data for paper` for more details!