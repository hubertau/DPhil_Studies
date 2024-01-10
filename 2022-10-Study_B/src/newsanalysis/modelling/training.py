from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from loguru import logger
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import numpy as np
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# define the compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    # calculate macro-averaged precision, recall, and F1 score
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # auc = roc_auc_score(labels, preds, multi_class='ovr')

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        # 'auc': auc
    }

# def compute_metrics(eval_pred):
#     metric = evaluate.load("accuracy")
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

def custom_trainer(
        dataset,
        checkpoint_dir,
        init_model = "sentence-transformers/LaBSE",
        num_train_epochs = 10,
        num_labels = 16,
        device = 'cpu',
        best_metric = 'accuracy',
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
        num_labels=num_labels
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
        num_train_epochs = num_train_epochs,
        metric_for_best_model=best_metric,
        disable_tqdm=True,
        remove_unused_columns=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()