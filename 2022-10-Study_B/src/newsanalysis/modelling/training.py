from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from loguru import logger
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def custom_trainer(
        dataset,
        checkpoint_dir,
        init_model = "sentence-transformers/LaBSE",
        num_train_epochs = 10,
        device = 'cpu',
        seed = 1
):
    # # load in dataset
    # ds = Dataset.load_from_disk(dataset)

    # # split into train test and dev
    # # 90% train, 10% test + validation
    # train_test = ds.train_test_split(test_size=0.1, seed = seed, shuffle=True)
    # # Split the 10% test + valid in half test, half valid
    # test_valid = train_test['test'].train_test_split(test=0.5, seed = seed, shuffle = True)
    # # gather everyone if you want to have a single DatasetDict
    # train_test_valid_dataset = DatasetDict(
    #     {
    #         'train': train_test['train'],
    #         'test': test_valid['test'],
    #         'valid': test_valid['train']
    #     }
    # )

    # load in datasetdict
    ds = DatasetDict.load_from_disk(dataset)

    # set up Trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        init_model,
        num_labels=16
    )

    if device == 'gpu':
        model = model.to('cuda')
        logger.info('Moved to GPU')

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_strategy = "epoch",
        num_train_epochs = num_train_epochs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['valid'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()