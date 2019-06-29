# README

The data and baseline codes for the [competition](https://biendata.com/competition/idiom/), which is adapted from the ACL 2019 paper "**[ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://arxiv.org/abs/1906.01265)**" (Zheng et al., 2019).

If you have any question, please get in touch with me zhengchj16@gmail.com.

## Competition Data

| Corpus             | Link                                                  |
| ------------------ | ----------------------------------------------------- |
| `train.txt`        | https://cloud.tsinghua.edu.cn/f/f8b048175a7b462cb2ba/ |
| `train_answer.csv` | https://cloud.tsinghua.edu.cn/f/0787d9e62f0e4992ad0e/ |
| `idiomDict.json`   | https://cloud.tsinghua.edu.cn/f/c5743ff2903445e2b8fc/ |
| `dev.txt`          | https://cloud.tsinghua.edu.cn/f/90c53b04a2374493acac/ |
| `dev_answer.csv`   | -                                                     |
| `test.txt`         | -                                                     |

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

`run.sh` is in the following format:

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

Results on other sets will be annouced later.

| Baselines | Dev      | Test | Out  |
| --------- | -------- | ---- | ---- |
| **BERT**  | 72.71305 | -    | -    |
| **AR**    | 65.40785 | -    | -    |

## Acknowledgement

Our codes are adapted from [PyTorch BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
