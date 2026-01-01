import torch
import yaml
import math
from pathlib import Path
from typing import Optional, Union
from transformers import T5Tokenizer

from model import Transformer

# all the configs 
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

BASE_DIR = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = load_config()
model_cfg, infer_cfg, path_cfg, train_cfg = (
    cfg["model"], cfg["inference"], cfg["paths"], cfg["training"]
)

DEFAULT_MODEL_PATH = (
    infer_cfg.get("checkpoint_path")
    or path_cfg.get("checkpoint")
    or train_cfg.get("checkpoint_path", "transformer_model.pt")
)
DEFAULT_MODEL_PATH = Path(DEFAULT_MODEL_PATH)
if not DEFAULT_MODEL_PATH.is_absolute():
    DEFAULT_MODEL_PATH = BASE_DIR / DEFAULT_MODEL_PATH

DEFAULT_TOKENIZER = infer_cfg["tokenizer_name"]
DEFAULT_MAX_LEN = model_cfg["max_len"]
DEFAULT_MAX_DECODE_LEN = infer_cfg["max_decode_len"]

PathLike = Union[str, Path]


# load the model 
def load_model(model_path: Optional[PathLike] = None):
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    checkpoint = torch.load(model_path, map_location=device)

    model = Transformer(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint["d_model"],
        num_heads=checkpoint["num_heads"],
        num_encoder_layers=checkpoint["num_encoder_layers"],
        num_decoder_layers=checkpoint["num_decoder_layers"],
        d_ff=checkpoint["d_ff"],
        max_len=checkpoint["max_len"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(f"model loaded from {model_path}")
    return model

# preprocess the input in the same way i preprocessed while training ... 
def format_table(table):
    if not isinstance(table, dict):
        return str(table)

    header = table.get("header", [])
    rows = table.get("rows", [])
    if not header:
        return table.get("name", "table")

    formatted = " | ".join(header)
    if rows:
        formatted += "\n" + "\n".join(" | ".join(map(str, row)) for row in rows[:10])
    return formatted


# generate sql 

def generate_sql(model, tokenizer, question, table, max_len=128, max_decode_len=50):
    table_str = format_table(table)
    # exact format 
    encoder_text = f"The question is: {question} table: {table_str}"
    
    
    print(f"Encoder input: {encoder_text[:200]}...")  # print first 200 chars of the input

    enc = tokenizer(
        encoder_text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    src = enc["input_ids"].to(device)
    dec = torch.tensor([[tokenizer.pad_token_id]], device=device)

    with torch.no_grad():
        src_mask = model.make_src_mask(src)
        src_emb = model.positional(model.embedding(src) * math.sqrt(model.embedding.d_model))
        enc_out = model.encoder(src_emb, src_mask)
        

        for step in range(max_decode_len):
            tgt_mask = model.make_tgt_mask(dec)
            tgt_emb = model.positional(model.embedding(dec) * math.sqrt(model.embedding.d_model))
            dec_out = model.decoder(tgt_emb, enc_out, src_mask, tgt_mask)

            logits = model.output_linear(dec_out)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            
            dec = torch.cat([dec, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(dec[0, 1:], skip_special_tokens=True)

class SqlInference:
    def __init__(self, model_path=None, tokenizer_name=None, max_len=None, max_decode_len=None):
        self.model = load_model(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(DEFAULT_TOKENIZER)
        self.max_len = max_len or DEFAULT_MAX_LEN
        self.max_decode_len = max_decode_len or DEFAULT_MAX_DECODE_LEN

    def predict(self, question, table, max_decode_len=None):
        return generate_sql(
            self.model,
            self.tokenizer,
            question,
            table,
            max_len=self.max_len,
            max_decode_len=max_decode_len or self.max_decode_len
        )


def inference_example():
    infer = SqlInference()

    
    question1 = "What is terrence ross' nationality"
    table1 = {
        "header": ["Player","No.","Nationality","Position","Years in Toronto","SchoolClub Team"],
        "rows": 
            [["Aleksandar Radojevi","25","Serbia","Center","1999-2000","Barton CC (KS)"],["Shawn Respert","31","United States","Guard","1997-98","Michigan State"],["Quentin Richardson","NA","United States","Forward","2013-present","DePaul"]]
    }
    print(f"Question: {question1}")
    print(f"Generated SQL: {infer.predict(question1, table1)}\n\n")
    
    
    question2 = "What position does Shawn Respert play"
    table2 = {
        "header": ["Player","No.","Nationality","Position","Years in Toronto","SchoolClub Team"],
        "rows": 
            [["Aleksandar Radojevi","25","Serbia","Center","1999-2000","Barton CC (KS)"],["Shawn Respert","31","United States","Guard","1997-98","Michigan State"],["Quentin Richardson","NA","United States","Forward","2013-present","DePaul"]]
    }
    print(f"Question: {question2}")
    print(f"Generated SQL: {infer.predict(question2, table2)}\n\n")
    
    question3 = "How many players are from United States"
    table3 = {
        "header": ["Player","No.","Nationality","Position","Years in Toronto","SchoolClub Team"],
        "rows": 
            [["Aleksandar Radojevi","25","Serbia","Center","1999-2000","Barton CC (KS)"],["Shawn Respert","31","United States","Guard","1997-98","Michigan State"],["Quentin Richardson","NA","United States","Forward","2013-present","DePaul"]]
    }
    print(f"Question: {question3}")
    print(f"Generated SQL: {infer.predict(question3, table3)}\n\n")
    
if __name__ == "__main__":
    
    inference_example()
