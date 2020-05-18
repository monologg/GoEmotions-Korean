from transformers import ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")

total_len = 0
max_len = 0
max_sent = 0
with open("data/test.tsv", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        sent = line.strip().split("\t")[0]
        tokens = tokenizer.tokenize(sent)
        total_len += len(tokens) + 2
        if max_len < len(tokens) + 2:
            max_len = len(tokens) + 2
            max_sent = line

    print("avg len: {:.2f}".format(total_len / len(lines)))
    print("max_len: {}".format(max_len))
    print(max_sent)
