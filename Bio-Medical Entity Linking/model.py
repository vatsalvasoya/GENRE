import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import AdamW
from tqdm import tqdm
import os


# 1. Setting CUDA device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU found, Using", device)


# 2. Loading the model
config = BartConfig.from_pretrained('models')
dropout_rate = 0.1
config.dropout = dropout_rate
model = (BartForConditionalGeneration.from_pretrained('models/', config=config)).to(device)
tokenizer = BartTokenizer.from_pretrained('GanjinZero/biobart-base')


# 3. Creating dataset class
class EntityDisambiguationDataset(Dataset):
    def __init__(self, input_text, target_labels, tokenizer):
        self.input_text = input_text
        self.target_labels = target_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        input_text = self.input_text[idx]
        target_label = self.target_labels[idx]

        # KEEPING THE SPECIAL TOKENS
        input_encoded = self.tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        input_ids = input_encoded["input_ids"].squeeze()
        attention_mask = input_encoded["attention_mask"].squeeze()

        target_encoded = self.tokenizer(target_label, return_tensors="pt", padding="max_length", truncation=True, max_length=60)
        target_ids = target_encoded["input_ids"].squeeze()

        return {
            "input_ids": (input_ids).to(device),
            "attention_mask": (attention_mask).to(device),
            "target_ids": (target_ids).to(device)
        }


# 4. Creating the dataloader
input_text = []
target_labels = []
with open("dataset/processed_data/train_dev_corpus.jsonl") as f:
    for line in f:
        data = json.loads(line)
        input_text.append(data["input"])
        target_labels.append(data["target"])



batch_size = 32
dataset = EntityDisambiguationDataset(input_text, target_labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Testing the dataloader
# iterator = iter(dataloader)
# print((next(iterator)["target_ids"]).tolist()[0])
# print(tokenizer.decode((next(iterator)["target_ids"]).tolist()[0]))



# 5. Creating the force words list

force_words_ids = [0, 2]
with open("dataset/processed_data/force_words.txt") as f:
    for word in f:
        force_word = word.strip()
        force_words_ids.extend(tokenizer(force_word, add_special_tokens=False).input_ids)

def prefix_allowed_tokens_fn(batch_id, input_ids):
    return force_words_ids

    

# 6. Training the model

optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]

        optimizer.zero_grad()

        # Use constrained decoding
        outputs = model.generate(
            input_ids,
            # num_beams=5,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            # forced_bos_token_id=tokenizer.bos_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
            # num_return_sequences=5,
            # no_repeat_ngram_size=1,
            # remove_invalid_values=True,
            # max_length=60
        )

        # Calculate loss
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).loss
        constrained_loss = model(input_ids=outputs,
                                #  attention_mask=attention_mask,
                                 labels=target_ids).loss

        total_loss = loss + constrained_loss

        total_loss.backward(retain_graph=True)
        # loss.backward(retain_graph=True)
        # constrained_loss.backward(retain_graph=True)
        
        optimizer.step()

        # print(tokenizer.decode(input_ids.tolist()[0]))
        # for i in range(len(outputs.tolist())):
        #     print("\n\n")
        #     print("input_ids: ", input_ids)
        #     print("predicted Entity: ", tokenizer.decode(outputs[i], skip_dpecial_tokens=True))
        #     print("Target Entity: ", tokenizer.decode(target_ids[i], skip_dpecial_tokens=True))
        #     print("predicted Entity: ", outputs[i])
        #     print("Target Entity: ", target_ids[i])
        #     print("\n\n")
        #     break
        # break

    # print("loss: ", loss.item())
    # print("constrained loss: ", constrained_loss.item())
    with open("loss.txt", "a") as f:
        f.write("{}\n".format(total_loss))
    print("total loss: ", total_loss.item())
    if (epoch+1+35)%5 == 0:
        os.mkdir(os.path.join("models/", "{}".format(epoch+1+35)))
        model.save_pretrained("models/{}".format(epoch+1+35))





