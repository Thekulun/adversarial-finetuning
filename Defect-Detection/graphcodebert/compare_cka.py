import argparse
import os

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torch_cka import CKA
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

# import sys
# sys.path.append("..")
from model import Model
from run import TextDataset


def compare_codebert(name1, name2, data_path, args):
    path_dict = {
        "ori": "./saved_models/ori",
        "adv": "./saved_models/FreeLB"
    }

    name_dict = {
        "ori": "GraphCodeBERT",
        "adv": "GraphCodeBERT-FreeLB",
        "pre": "GraphCodeBERT-pre"
    }



    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    model1 = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model1 = Model(model1, config, tokenizer, None)

    if name1 != "pre":
        model_path1 = path_dict[name1]
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(model_path1, '{}'.format(checkpoint_prefix))
        model1.load_state_dict(torch.load(output_dir))


    model1.to(args.device)

    # for name, param in model1.named_parameters():
    #     print(name, param.shape)
    #
    # exit(0)

    model2 = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model2 = Model(model2, config, tokenizer, None)

    if name2 != "pre":
        model_path2 = path_dict[name2]
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(model_path2, '{}'.format(checkpoint_prefix))
        model2.load_state_dict(torch.load(output_dir))
    model2.to(args.device)

    test_dataset = TextDataset(tokenizer, args, data_path)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=8, num_workers=4,
                                 pin_memory=True)

    layers = ['encoder.roberta.embeddings']
    for i in range(12):
        layer = 'encoder.roberta.encoder.layer.{}.output'.format(i)
        layers.append(layer)

    # layers.append('encoder.classifier')

    cka = CKA(model1, model2,
              model1_name=name_dict[name1],  # good idea to provide names to avoid confusion
              model2_name=name_dict[name2],
              model1_layers=layers,
              model2_layers=layers,
              device='cuda')
    cka.compare(test_dataloader)  # secondary dataloader is optional
    if not os.path.exists("./cka/"):
        os.makedirs("./cka/")
    cka.plot_results(save_path=f"./cka_benign/Defect_FreeLB_GraphCodeBERT_{name1}_{name2}.png")
    results = cka.export()  # returns a dict that contains model names, layer names and the CKA matrix

    for key, value in results.items():
        print(key, value)


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--model_name_or_path", default=None, type=str,
                    help="The model checkpoint for weights initialization.")

parser.add_argument("--test_data_file", default="../dataset/test.jsonl", type=str,
                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
parser.add_argument("--block_size", default=-1, type=int,
                    help="Optional input sequence length after tokenization."
                         "The training dataset will be truncated in block of this size for training."
                         "Default to the model max input length for single sentence inputs (take into account special tokens).")


args = parser.parse_args()

#args.block_size = 512
args.code_length = 448
args.data_flow_length = 64

args.model_name_or_path = "microsoft/graphcodebert-base"
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.test_data_file = "/home/fdse/sjw/ISSTA22-CodeStudy/Task/Defect-Detection/dataset/test.jsonl"

name1 = "ori"
name2 = "adv"
name3 = "pre"


compare_codebert(name1, name2, args.test_data_file, args)



# for name, param in model1.named_parameters():
#     print(name, param.shape)
