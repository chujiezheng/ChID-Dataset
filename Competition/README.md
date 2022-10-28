# README

The data and baseline codes for the [competition](https://biendata.com/competition/idiom/), which is adapted from the ACL 2019 paper "**[ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://www.aclweb.org/anthology/P19-1075)**" (Zheng et al., 2019).

## Competition Data

### Download Link

[Here](https://1drv.ms/u/s!Aky8v8NZbQx1qjter8p5SKFs8GtY?e=hssZZx).

### Data Description

One example is shown below (from the training data):

```python
{
  "content": [
    # passage 0
    "……在热火22年的历史中，他们已经100次让对手得分在80以下，他们在这100次中都取得了胜利，今天他们希望能#idiom000378#再进一步。", 
    # passage 1
    "在轻舟发展过程之中，是和业内众多企业那样走相似的发展模式，去#idiom000379#？还是迎难而上，另走一条与众不同之路。诚然，#idiom000380#远比随大流更辛苦，更磨难，更充满风险。但是有一条道理却是显而易见的：那就是水往低处流，随波逐流，永远都只会越走越低。只有创新，只有发展科技，才能强大自己。", 
    # passage 2
    "最近十年间，虚拟货币的发展可谓#idiom000381#。美国著名经济学家林顿·拉鲁什曾预言：到2050年，基于网络的虚拟货币将在某种程度上得到官方承认，成为能够流通的货币。现在看来，这一断言似乎还嫌过于保守……", 
    # passage 3
    "“平时很少能看到这么多老照片，这次图片展把新旧照片对比展示，令人印象深刻。”现场一位参观者对笔者表示，大多数生活在北京的人都能感受到这个城市#idiom000382#的变化，但很少有人能具体说出这些变化，这次的图片展按照区域发展划分，展示了丰富的信息，让人形象感受到了60年来北京的变化和发展。", 
    # passage 4
    "从今天大盘的走势看，市场的热点在反复的炒作之中，概念股的炒作#idiom000383#，权重股走势较为稳健，大盘今日早盘的震荡可以看作是多头关前的蓄势行为。对于后市，大盘今日蓄势震荡后，明日将会在权重和题材股的带领下亮剑冲关。再创反弹新高无悬念。", 
    # passage 5
    "……其中，更有某纸媒借尤小刚之口指出“根据广电总局的这项要求，2009年的荧屏将很难出现#idiom000384#的情况，很多已经制作好的非主旋律题材电视剧想在卫视的黄金时段播出，只能等到2010年了……"],
  "candidates": [
    "百尺竿头", 
    "随波逐流", 
    "方兴未艾", 
    "身体力行", 
    "一日千里", 
    "三十而立", 
    "逆水行舟", 
    "日新月异", 
    "百花齐放", 
    "沧海一粟"
  ]
}
```

The corresponding answers are below (the second column indicates indexs of the golden truths in the candidate list):

```
#idiom000378#,0
#idiom000379#,1
#idiom000380#,6
#idiom000381#,4
#idiom000382#,7
#idiom000383#,2
#idiom000384#,8
```

It can be observed that, for the blank `#idiom000381#` in the passage 2, the three idioms "方兴未艾", "一日千里", "日新月异" in the candidates are all in line with the context and share similar meanings. However, considering the blank `#idiom000382#` in the passage 3 in which only the idiom "日新月异" can be filled, and that "方兴未艾" is the only suitable one for the blank `#idiom000383#` in the passage 4, we should fill "一日千里" into `#idiom000381#` according to the exclusion method.

In general, the main differences of ChID-Adapt data for the competition from the original ChID dataset are following:

- It adopts a new type of problem. A list of passages (not an isolated one) is provided and the answers need to be selected from a given set of candidate idioms with fixed length. Under this setting, more strategies may be allowed (for instance, the exclusion method).
- It establishes up connections between blanks. In each data, blanks share the same candidate list and their golden answers tend to have similar meanings. It is required to understand the meaning of idioms, distinguish their differences and compare similar contexts to make the correct decision.

## Requirements

- Python 3.5
- For  `RNN-based Baseline`
  - Tensorflow 1.4
  - Jieba
- For  `BERT-based Baseline`
  - PyTorch 1.0

## RNN-based Baseline

The RNN-based baseline ([Attentive Reader](https://arxiv.org/abs/1506.03340)) is consistent with that described in the paper. To run it, please refer to `Codes for baseline` for details.

## BERT-based Baseline

We also provide a simple BERT-based baseline system (PyTorch version) for participants.

### Preparation

Our codes are adapted from [PyTorch BERT](https://github.com/huggingface/pytorch-pretrained-BERT). The baseline is based on a simple multi-choice framework, which matches the embeddings of candidate idioms with the output vector at the blank position.

With codes downloaded, you have to get pre-trained Chinese BERT for initialization purpose. Here we make use of the pre-trained model [Chinese BERT with Whole Word Masking](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD) provided by Cui Y et al., which may bring improvement for Chinese tasks.

### Training and Predicting

We assume that all the files are placed in the correct path. 

Pre-trained Chinese BERT weights (PyTorch version) should be placed in `chinese_wwm_pytorch` folder. Then you can run the `run.sh` in the command line:

```
bash run.sh
```

`run.sh` is in the following format (also the default setting we used to train the BERT baseline):

```bash
python run_chid.py \
	--vocab_file ./chinese_wwm_pytorch/vocab.txt \
	--bert_config_file ./chinese_wwm_pytorch/bert_config.json \
	--init_checkpoint ./chinese_wwm_pytorch/pytorch_model.bin \
	--do_train \
	--do_predict \
	--train_file ../data/train.txt \
	--train_ans_file ../data/train_answer.csv \
	--predict_file ../data/dev.txt \
	--train_batch_size 32 \
	--predict_batch_size 128 \
	--learning_rate 2e-5 \
	--num_train_epochs 10.0 \
	--max_seq_length 256 \
	--output_dir ./output_model
```

If you have successfully trained your model, you could simply remove the line of `--do_train \` for the prediction on the development set. Then your prediction results will be generated as the file `./output_model/prediction.csv`.

## Baseline Results

| Baselines | Dev      | Test | Out  |
| --------- | -------- | ---- | ---- |
| **BERT**  | 72.71305 | 72.36848    | 64.64770    |
| **AR**    | 65.40785 | 65.57370    | 55.64539    |

Results on other sets will be annouced later.

## Acknowledgement

Our codes are adapted from [PyTorch BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and the pre-trained Chinese BERT model is from [Chinese BERT with Whole Word Masking](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD).
