# README

Baseline code for the ACL 2019 paper "**[ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://arxiv.org/abs/1906.01265)**" (Zheng et al., 2019)

If you have any question, please get in touch with me zhengchj16@gmail.com

## Requirements

- Python 3.5
- Tensorflow 1.4
- Jieba

## Run

The default file paths are listed as following:

- The directory of data: "../data", where "train_data.txt" and "dev_data.txt" should be placed
- The list of idioms: "./idiomList.txt", which can be loaded by `eval(f.readline())`
- The list of words: "./wordList.txt", where the first 100,000 words are held
  - This means that you need to get the list of words based on the training data by yourself

- Pretrained embeddings of idioms: "../data/idiomvector.txt" without the header
- Pretrained embeddings of words: "../data/wordvector.txt" without the header

The file paths are defined in `DataManager.py` and `utils.py`, and can be reset by yourself

You can run the program by the following command

```bash
python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Available options can be found in `Flags.py`.

For example, to train the model [Stanford Reader](https://www.aclweb.org/anthology/P16-1223) with default settings:

```bash
python main.py --model sar
```