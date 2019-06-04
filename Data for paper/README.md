# README

The download links of the corpus for the paper are listed here.

## Download Links

|   Corpus   |   Link   |
| ---- | ---- |
|   **Train**   |   https://cloud.tsinghua.edu.cn/f/38c307945a884e56b519/   |
|   **Dev**   |   It will be available later.   |


## Data Description

One example is shown below:

```python
{
  "content": "由实力派演员刘威饰演的清华第三任校长蒋南翔，是我国著名的青年运动家和教育家，他跟清华终身校长梅贻琦一样，都是由清华人自己培养出来的校长。历史上的蒋南翔是著名的“一二九”学生救亡运动的领导人之一，他在清华校长之位14年期间，不但很好的继承了清华建校之初的优秀传统与理念，而且更加的#idiom#，他把清华的教师队伍扩大了将近5倍，将清华本科人数破万，为新中国培养了大量的有用人才。在《天行健》中饰演蒋南翔的刘威是观众所熟悉的著名实力派演员，早在1987年刘威就在《关东大侠》中饰演豪爽仗义的关云天一角而获得了金鸡奖最佳男主角的提名，后来更是因在《唐明皇》中精湛的表演而一举夺得金鹰奖最佳男演员奖。此次《天行健》选定刘威来出演正是看中了他#idiom#的表演方式和对人物深入内心的刻画。至此，《天行健》中涉及的三位清华校长的人选都已经曝光，#idiom#的第一任校长赵文?、稳重坚毅的第二任校长孙逊、亲切务实的第三任校长刘威，再加上梁思成、林徽因、朱自清、闻一多等一批“大师”的加盟，相信作为清华百年校庆重点项目之一的《天行健》一定会带领观众重温那段不能抹去的历史。", 
  "realCount": 3,
  "groundTruth": ["发扬光大", "平易近人", "温文尔雅"], 
  "candidates": [
    ["意气风发", "街谈巷议", "人才辈出", "一脉相传", "后继有人", "发扬光大", "腥风血雨"], 
    ["平易近人", "落落大方", "八仙过海", "彬彬有礼", "史无前例", "盛气凌人", "好自为之"], 
    ["不拘小节", "风流潇洒", "无病呻吟", "言谈举止", "壮志凌云", "关门闭户", "温文尔雅"]
  ]
}
```

- `content`: The given passage where the original idioms are replaced by placeholders `#idiom#`
- `realCount`: The number of placeholders or blanks
- `groundTruth`: The golden answers in the order of blanks
- `candidates`: The given candidates in the order of blanks
