import os
import json
import re


def preprocess_data(raw_data_dir, processed_data_dir, raw_data_file, processed_data_file, entity_file, force_words_file, force_seq_file):
    entity_list = dict()
    force_words = set()
    with open(os.path.join(raw_data_dir, entity_file), encoding='utf-8') as f:
        for line in f:
            index = line.find("\t")
            if index != -1:
                entity_id = line[:index].strip()
                entity = line[index+1:].strip()
                for word in entity.split():
                    force_words.add(word)
                entity_list[entity_id] = entity

    regex = re.compile('^\d+\|[a|t]\|')
    
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    documents = dict()
    annotated_docs = []
    count = 0
    start_token = "[START_ENT] "
    end_token = " [END_ENT]"

    with open(os.path.join(raw_data_dir, raw_data_file), encoding='utf-8') as f:
        for line in f:
            doc = dict()
            line = line.strip()
            if regex.match(line):
                match_span = regex.match(line).span()
                start_span_idx = match_span[0]
                end_span_idx = match_span[1]

                document_id = line[start_span_idx:end_span_idx].split("|")[0]
                text = line[end_span_idx:]

                if document_id not in documents:
                    documents[document_id] = text
                else:
                    documents[document_id] = documents[document_id] + ' ' + text

            else:
                cols = line.strip().split('\t')
                if len(cols) == 6:
                    if cols[5] == '-1':
                        continue
                    document_id = cols[0]
                    
                    start_index = int(cols[1])
                    end_index = int(cols[2])
                    entity_id = cols[5]

                    if entity_id in entity_list:  # TODO: REMOVE THIS TO INCORPORATE ALL THE ENTITIES: "C009166" this ID gave a key error
                        doc["id"] = str(document_id) + "_" + str(count)
                        count += 1
                        
                        input_text = documents[document_id]
                        input_text = input_text[:end_index] + end_token + input_text[end_index:]
                        input_text = input_text[:start_index] + start_token + input_text[start_index:]

                        doc["input"] = input_text
                        doc["target"] = entity_list[entity_id]

                        annotated_docs.append(doc)
    
    with open(os.path.join(processed_data_dir, processed_data_file), 'w') as f:
        for item in annotated_docs:
            f.write(json.dumps(item) + "\n")

    
    with open(os.path.join(processed_data_dir, force_words_file), "w") as f:
        for item in force_words:
            f.write("%s\n" % item)


    with open(os.path.join(processed_data_dir, force_seq_file), "w") as f:
        for item in entity_list.values():
            f.write("%s\n" % item)

    
                    


if __name__ == "__main__":
    preprocess_data("dataset/raw_data/", "dataset/processed_data/", "test_corpus.txt", "test_corpus.jsonl", "entities.txt", "force_words.txt", "force_sequences.txt")
