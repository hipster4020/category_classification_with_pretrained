import hydra
import numpy as np
from datasets import load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

import wandb
from dataloader import load

wandb.init(project="em_category")


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    print("tokenizer start...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.pretrained_model_name)
    print("tokenizer end...")

    # data loder
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    # model
    print("model start...")
    args = TrainingArguments(
        **cfg.TRAININGS,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL.pretrained_model_name,
        num_labels=cfg.MODEL.num_labels,
    )
    print("model end...")

    def sigmoid(x):
        x = 1 / (1 + np.exp(-x))
        return x

    # metrics
    def compute_metrics(eval_preds):
        metric = load_metric(cfg.METRICS.metric_name)
        logits, labels = eval_preds

        # category predict
        x = sigmoid(logits)
        logits = np.where(x > 0.5, 1.0, 0.0)

        return metric.compute(
            predictions=logits.reshape(-1),
            references=labels.reshape(-1),
            average=cfg.METRICS.average,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(cfg.PATH.save_dir)


if __name__ == "__main__":
    main()
