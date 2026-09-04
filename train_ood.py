import argparse
import json
import os
import pickle
import warnings
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn import svm

from src.ood_utils import set_seed, collate_fn
from src.ood_model_selector import CodeBERTForSelector
from src.ood_data import load_dforget

warnings.filterwarnings("ignore")
torch.set_num_threads(10)


def fit_ocsvm(args, scores, groups):
    """Fit one one-class SVM per deprecated API on its score vectors s(x)."""
    index_by_api = defaultdict(list)
    for i, apis in enumerate(groups):
        for api in apis:
            index_by_api[api].append(i)

    ocsvm, group_sizes = {}, {}
    for api in sorted(index_by_api):
        idx = index_by_api[api]
        clf = svm.OneClassSVM(nu=args.nu, kernel=args.kernel)
        clf.fit(scores[idx])
        ocsvm[api] = clf
        group_sizes[api] = len(idx)
        print("  {}: {} samples, {} SV".format(api, len(idx), clf.support_vectors_.shape[0]))
    return ocsvm, group_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="tummitum/codebert-deprecated", type=str)
    parser.add_argument("--data_path", default="./data/codellama/D_forget.json", type=str)
    parser.add_argument("--output_dir", default="./ckpt", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
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

    # 2) score vector s(x) per record: Mahalanobis + (minus) max cosine, one dim per layer
    scores = model.get_unsup_Mah_score(dataloader, mean_list, precision_list, fea_list)
    print("score matrix:", scores.shape)

    # 3) one one-class SVM per deprecated API
    ocsvm, group_sizes = fit_ocsvm(args, scores, groups)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"mean_list": mean_list, "precision_list": precision_list, "fea_list": fea_list},
               os.path.join(args.output_dir, "stats.pt"))
    with open(os.path.join(args.output_dir, "ocsvm.pkl"), "wb") as f:
        pickle.dump(ocsvm, f)
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({"nu": args.nu, "kernel": args.kernel, "num_apis": len(ocsvm),
                   "group_sizes": group_sizes}, f, indent=2)
    print("Saved {} one-class SVMs to {}".format(len(ocsvm), args.output_dir))


if __name__ == "__main__":
    main()
