import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, required=True)
parser.add_argument('--answer_file', type=str, required=True)
args = parser.parse_args()

pred = open(args.pred_file).readlines()
ans = open(args.answer_file).readlines()
assert len(pred) == len(ans)

ans_dict = {}
pred_dict = {}
for line in ans:
    line = line.strip().split(',')
    ans_dict[line[0]] = int(line[1])
for line in pred:
    line = line.strip().split(',')
    pred_dict[line[0]] = int(line[1])

cnt = 0
acc = 0
for key in ans_dict:
    assert key in pred_dict
    cnt += 1
    if ans_dict[key] == pred_dict[key]:
        acc += 1

print(acc / cnt * 100)
