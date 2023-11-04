from transformers import BartTokenizer, BartForConditionalGeneration
import torch

from typing import List
from trie import MarisaTrie, Trie



tokenizer = BartTokenizer.from_pretrained('GanjinZero/biobart-base')
model = BartForConditionalGeneration.from_pretrained('models/').eval()


force_seq = []
max_len = 0
with open("dataset/processed_data/force_sequences.txt") as f:
    for line in f:
        # l = len(tokenizer.encode(line.strip(), add_special_tokens=False))
        # if(l > max_len):
        #     max_len = l
        force_seq.append(tokenizer.encode(line.strip()))
    
# print(max_len)


trie = Trie(force_seq)

def prefix_allowed_tokens_fn(batch_id, input_ids):
    # print(tokenizer.decode(input_ids))
    # print(trie.get(input_ids.tolist()))
    return trie.get(input_ids.tolist())



# # CHECKING THE VALIDITY OF TRIE

# input_ids = tokenizer.encode("Heart", return_tensors="pt", add_special_tokens=False)
# # print(tokenizer.decode(input_ids[0]))
# outputs = prefix_allowed_tokens_fn(1, input_ids[0])
# for output in outputs:
#     print(tokenizer.decode(output))













# input = "Linezolid-induced [START_ENT] optic neuropathy [END_ENT]. Many systemic antimicrobials have been implicated to cause ocular adverse effects. This is especially relevant in multidrug therapy where more than one drug can cause a similar ocular adverse effect. We describe a case of progressive loss of vision associated with linezolid therapy. A 45-year-old male patient who was on treatment with multiple second-line anti-tuberculous drugs including linezolid and ethambutol for extensively drug-resistant tuberculosis (XDR-TB) presented to us with painless progressive loss of vision in both eyes. Color vision was defective and fundus examination revealed optic disc edema in both eyes. Ethambutol-induced toxic optic neuropathy was suspected and tablet ethambutol was withdrawn. Deterioration of vision occurred despite withdrawal of ethambutol. Discontinuation of linezolid resulted in marked improvement of vision. Our report emphasizes the need for monitoring of visual function in patients on long-term linezolid treatment."
# TARGET = Optic Nerve Diseases

# input = "Late-onset  [START_ENT] scleroderma renal crisis [END_ENT]  induced by tacrolimus and prednisolone: a case report. Scleroderma renal crisis (SRC) is a rare complication of systemic sclerosis (SSc) but can be severe enough to require temporary or permanent renal replacement therapy. Moderate to high dose corticosteroid use is recognized as a major risk factor for SRC. Furthermore, there have been reports of thrombotic microangiopathy precipitated by cyclosporine in patients with SSc. In this article, we report a patient with SRC induced by tacrolimus and corticosteroids. The aim of this work is to call attention to the risk of tacrolimus use in patients with SSc."
# TARGET = Kidney Diseases

# input = "Co-carcinogenic effect of retinyl acetate on [START_ENT] forestomach carcinogenesis [END_ENT] of male F344 rats induced with butylated hydroxyanisole. The potential modifying effect of retinyl acetate (RA) on butylated hydroxyanisole (BHA)-induced rat forestomach tumorigenesis was examined. Male F344 rats, 5 weeks of age, were maintained on diet containing 1% or 2% BHA by weight and simultaneously on drinking water supplemented with RA at various concentrations (w/v) for 52 weeks. In groups given 2% BHA, although marked hyperplastic changes of the forestomach epithelium were observed in all animals, co-administration of 0.25% RA significantly (P less than 0.05) increased the incidence of forestomach tumors (squamous cell papilloma and carcinoma) to 60% (9/15, 2 rats with carcinoma) from 15% (3/20, one rat with carcinoma) in the group given RA-free water. In rats given 1% BHA, RA co-administered at a dose of 0.05, 0.1, 0.2 or 0.25% showed a dose-dependent enhancing effect on the development of the BHA-induced epithelial hyperplasia. Tumors, all papillomas, were induced in 3 rats (17%) with 0.25% RA and in one rat (10%) with 0.05% RA co-administration. RA alone did not induce hyperplastic changes in the forestomach. These findings indicate that RA acted as a co-carcinogen in the BHA forestomach carcinogenesis of the rat."
# TARGET = Stomach Neoplasms

# input = "Acute [START_ENT] hepatitis [END_ENT] associated with clopidogrel: a case report and review of the literature. Drug-induced hepatotoxicity is a common cause of acute hepatitis, and the recognition of the responsible drug may be difficult. We describe a case of clopidogrel-related acute hepatitis. The diagnosis is strongly suggested by an accurate medical history and liver biopsy. Reports about cases of hepatotoxicity due to clopidogrel are increasing in the last few years, after the increased use of this drug. In conclusion, we believe that physicians should carefully consider the risk of drug-induced hepatic injury when clopidogrel is prescribed."
# TARGET = Chemical and Drug Induced Liver Injury

# input = "Lidocaine-induced [START_ENT] cardiac asystole [END_ENT]. Intravenous administration of a single 50-mg bolus of lidocaine in a 67-year-old man resulted in profound depression of the activity of the sinoatrial and atrioventricular nodal pacemakers. The patient had no apparent associated conditions which might have predisposed him to the development of bradyarrhythmias; and, thus, this probably represented a true idiosyncrasy to lidocaine."
# TARGET = Heart Arrest

input_ids = tokenizer.encode(input, return_tensors="pt", add_special_tokens=False)


# print(tokenizer.decode(input_ids[0]))


outputs = model.generate(
    inputs=input_ids,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    decoder_start_token_id=tokenizer.bos_token_id,
    num_beams=1,
    num_return_sequences=1,
    # no_repeat_ngram_size=1,
    # remove_invalid_values=True,
    # max_length=10
)


for i, output in enumerate(outputs):
    # print(tokenizer.decode(output))
    print(i, ": ", tokenizer.decode(output, skip_special_tokens=True))