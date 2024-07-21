import torch

from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer

from tqdm.auto import tqdm
model_name = "Rostlab/prot_bert"
#from utility.model_util import get_device

if "t5" in model_name:
  tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
  model = T5EncoderModel.from_pretrained(model_name)
elif "albert" in model_name:
  tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = AlbertModel.from_pretrained(model_name)
elif "bert" in model_name:
  tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = BertModel.from_pretrained(model_name)
elif "xlnet" in model_name:
  tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = XLNetModel.from_pretrained(model_name)
else:
  print("Unkown model name")

device = "cpu"
model = model.to(device)
# Remove any special tokens after embedding
if "t5" in model_name:
    shift_left = 0
    shift_right = -1
elif "bert" in model_name:
    shift_left = 1
    shift_right = -1
elif "xlnet" in model_name:
    shift_left = 0
    shift_right = -2
elif "albert" in model_name:
    shift_left = 1
    shift_right = -1
else:
    print("Unkown model name")

def embed_dataset(dataset_seqs):
    inputs_embedding = {}
    for sample in tqdm(dataset_seqs):
        with torch.no_grad():
            ids = tokenizer.batch_encode_plus([c for c in sample], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            inputs_embedding[sample] = embedding[0].detach().cpu().numpy()[shift_left:shift_right]

    return inputs_embedding