import os

import hydra
import torch
from pshmodule.utils import filemanager as fm
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer & model
    print(f"load tokenizer...", end=" ")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.pretrained_model_name)
    print("tokenizer loading done!")

    # model
    print(f"load model...", end=" ")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.PATH.save_dir,
        num_labels=cfg.MODEL.num_labels,
    )
    model.cuda().eval()
    print("load model done!")

    # data load
    df = fm.load(cfg.PATH.untrain_data_path)
    df.dropna(axis=0, inplace=True)
    print(df.head())
    print(df.shape)

    # predict
    pred_list = []
    for sentence in tqdm(df.content):
        data = tokenizer(
            sentence,
            max_length=cfg.DATASETS.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            data = {k: v.cuda() for k, v in data.items()}
            outputs = model(output_hidden_states=True, **data)

            # category predict
            x = torch.sigmoid(outputs.logits[0]).double().to("cpu")
            logits = torch.where(x > 0.5, 1.0, 0.0)

            category = ""
            zero_count = 0
            for i in logits:
                if i.item() == 0.0:
                    zero_count += 1

            # when there is no category.
            if zero_count == 30:
                category = "etc"
            # when there is category.
            else:
                predict = [
                    cfg.DICT.labels[k] for k, v in enumerate(logits) if v.item() == 1.0
                ]
                category = ", ".join(predict)

            pred_list.append(category)

    # making label data
    df["predict"] = pred_list
    print(f"df length : {len(df)}")

    # data save
    print("data save start...")
    fm.save(cfg.PATH.data_dir, df)
    print("data save end...")


if __name__ == "__main__":
    main()
