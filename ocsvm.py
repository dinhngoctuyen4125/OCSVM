import argparse
import json
import os
import pickle
import warnings


import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn import svm

from src.ood_utils import set_seed, collate_fn
from src.ood_model_selector import CodeBERTForSelector
from src.ood_data import load_dforget

warnings.filterwarnings("ignore")
torch.set_num_threads(10)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="tummitum/codebert-deprecated", type=str)
    parser.add_argument("--data_path", default="./data/codellama/D_forget.json", type=str)
    parser.add_argument("--output_dir", default="./ckpt", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--kernel", type=str, default="linear")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = CodeBERTForSelector(args.model_name_or_path, device=device)
    model.to(device)

    dataset, groups = load_dforget(args.data_path, tokenizer, max_seq_length=args.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    print("Loaded {} records from {}".format(len(dataset), args.data_path))

    # 1) 13-layer statistics of D_forget: centroid, precision, normalised feature bank
    mean_list, precision_list, fea_list = model.sample_X_estimator(dataloader)

    # 2) score vector s(x) per record (drop layer 0 like original)
    scores = model.get_unsup_Mah_score(dataloader, mean_list, precision_list, fea_list)[:, 1:]
    print("score matrix:", scores.shape)

    # 3) fit one OCSVM on all D_forget
    c_lr = svm.OneClassSVM(nu=args.nu, kernel=args.kernel)
    c_lr.fit(scores)
    print("OCSVM: {} samples, {} SVs".format(scores.shape[0], c_lr.support_vectors_.shape[0]))

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"mean_list": mean_list, "precision_list": precision_list, "fea_list": fea_list},
               os.path.join(args.output_dir, "stats.pt"))
    with open(os.path.join(args.output_dir, "ocsvm.pkl"), "wb") as f:
        pickle.dump(c_lr, f)
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({"nu": args.nu, "kernel": args.kernel,
                   "num_samples": int(scores.shape[0])}, f, indent=2)
    print("Saved OCSVM to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
