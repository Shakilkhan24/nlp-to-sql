
import torch
import json
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class WikiSQLDataset(Dataset):
    def __init__(self, path, tokenizer_name="t5-small", max_len=128):
        self.samples = []
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len


        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"the json is not rightly formatted {line_num}: {e}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        question = item["question"]
        sql = item["sql"]["human_readable"]
        
        # converting table to string and a nice format 
        if isinstance(item["table"], dict):
            table_dict = item["table"]
            header = table_dict.get("header", [])
            rows = table_dict.get("rows", [])
            
            if header:  # columns 
                table_str = " | ".join(str(h) for h in header)
                if rows:
                    table_str += "\n" + "\n".join(" | ".join(str(cell) for cell in row) for row in rows[:10])  # only first 10 rows
            else:
                table_str = str(table_dict.get("name", "table"))     # if table not given, then just use the table name
            table = table_str
        else:
            table = str(item["table"])

        # Encoder input
        encoder_text = f"The question is: {question} table: {table}"

        encoder = self.tokenizer(
            encoder_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        
        # suppose max_len=6:
        # decoder input tokens: [tok1, tok2, tok3, eos, pad, pad]
        # Decoder input:  [pad, tok1, tok2, tok3, eos, pad]
        # Labels: [tok1, tok2, tok3, eos, pad, pad] -> [tok1, tok2, tok3, eos, -100, -100]
        
        # tokenize the SQL query
        decoder_tokens = self.tokenizer(
            sql,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        
        token_ids = decoder_tokens["input_ids"].squeeze(0)  # (max_len,)
        attention_mask = decoder_tokens["attention_mask"].squeeze(0)  # (max_len,)
        
        decoder_input_ids = torch.cat([
            torch.tensor([self.tokenizer.pad_token_id]),  # Start token
            token_ids[:-1]  # All tokens except the last one
        ])
        
        # labels are the original tokens
        labels = token_ids.clone()
        
        # set all padding tokens as -100 
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        decoder_attention_mask = torch.cat([
            torch.tensor([1]),  # For the starting pad_token (decoder start)
            attention_mask[:-1]
        ])

        return {
            "encoder_input_ids": encoder["input_ids"].squeeze(0),
            "encoder_attention_mask": encoder["attention_mask"].squeeze(0),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels
        }

