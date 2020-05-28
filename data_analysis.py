from tokenization_kocharelectra import KoCharElectraTokenizer


tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")

unk_vocab = set()


###################
max_len = 0
avg_len = 0

with open("data/train.tsv", 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        tokens = tokenizer.tokenize(line)
        for token in tokens:
            if token not in tokenizer.vocab:
                unk_vocab.add(token)
        token_len = len(tokens) + 2
        max_len = max(max_len, token_len)
        avg_len += token_len

avg_len /= len(lines)

print("avg_len: {:.2f}".format(avg_len))
print("max_len: {}".format(max_len))

###################
max_len = 0
avg_len = 0

with open("data/dev.tsv", 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        tokens = tokenizer.tokenize(line)
        for token in tokens:
            if token not in tokenizer.vocab:
                unk_vocab.add(token)
        token_len = len(tokens) + 2
        max_len = max(max_len, token_len)
        avg_len += token_len

avg_len /= len(lines)

print("avg_len: {:.2f}".format(avg_len))
print("max_len: {}".format(max_len))

###################
max_len = 0
avg_len = 0

with open("data/test.tsv", 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        tokens = tokenizer.tokenize(line)
        for token in tokens:
            if token not in tokenizer.vocab:
                unk_vocab.add(token)
        token_len = len(tokens) + 2
        max_len = max(max_len, token_len)
        avg_len += token_len

avg_len /= len(lines)

print("avg_len: {:.2f}".format(avg_len))
print("max_len: {}".format(max_len))


print(unk_vocab)

with open("unk_vocab.txt", "w", encoding="utf-8") as f:
    for token in unk_vocab:
        f.write("{}\n".format(token))