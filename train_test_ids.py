import os


all_sequence_txt = "dataset/AnimalMotion/all_sequence_dirs.txt"

all_sequence_ids = []
with open(all_sequence_txt, "r") as f:
    lines = f.readlines()
    for line in lines:
        seq_id = os.path.basename(line.strip())
        all_sequence_ids.append(seq_id)

train_txt = "dataset/AnimalMotion/train.txt"
test_txt = "dataset/AnimalMotion/test.txt"
all_sequence_txt_test = "dataset/AnimalMotion_test/all_sequence_dirs.txt"
trian_ids = []
test_ids = []

with open(all_sequence_txt_test, "r") as f:
    lines = f.readlines()
    for line in lines:
        seq_id = os.path.basename(line.strip())
        test_ids.append(seq_id)


train_ids = list(set(all_sequence_ids) - set(test_ids))
with open(train_txt, "w") as f:
    for seq_id in train_ids:
        f.write(seq_id + "\n")
with open(test_txt, "w") as f:
    for seq_id in test_ids:
        f.write(seq_id + "\n")

