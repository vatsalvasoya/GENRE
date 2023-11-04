from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import json
from trie import Trie
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU found, Using", device)
# device = "cpu"

tokenizer = BartTokenizer.from_pretrained('GanjinZero/biobart-base')
model = (BartForConditionalGeneration.from_pretrained('models/').eval()).to(device)


force_seq = []
max_len = 0
with open("dataset/processed_data/force_sequences.txt") as f:
    for line in f:
        force_seq.append(tokenizer.encode(line.strip()))
    


trie = Trie(force_seq)

def prefix_allowed_tokens_fn(batch_id, input_ids):
    return trie.get(input_ids.tolist())


input_text = []
target_labels = []
with open("dataset/processed_data/test_corpus.jsonl") as f:
    for line in f:
        data = json.loads(line)
        input_text.append(data["input"])
        target_labels.append(data["target"])


correct_samples = 0
total_samples = len(input_text)
num_beams = 1
K = 1
for input, target in tqdm(zip(input_text, target_labels), desc="Processing "):
    input_ids = (tokenizer.encode(input, return_tensors="pt", add_special_tokens=False)).to(device)

    outputs = model.generate(
        inputs=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        decoder_start_token_id=tokenizer.bos_token_id,
        num_beams=num_beams,
        num_return_sequences=num_beams,
    )

    for i in range(0, min(K, len(outputs))):
        output = tokenizer.decode(outputs[i], skip_special_tokens=True)
        if(output == target):
            correct_samples += 1
            break

    
print("Correct samples: ", correct_samples)
print("Total samples: ", total_samples)
print("Recall@{}: {}".format(K, (correct_samples/total_samples)*100))


