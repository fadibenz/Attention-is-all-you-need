from architectures.Transformer import Transformer
from data.make_dataset import make_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

if __name__ == '__main__':

    dataset_train, dataset_validation, dataset_test, data_collator, tokenizer = make_dataset()

    model = Transformer(
        tokenizer.vocab_size,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        max_len=1024,
        n_layers=4,
        d_model=512,
        n_heads=8,
        d_ffn=2048,
        p_drop=0.1
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="../models",

        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,

        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        dataloader_drop_last=True,
        per_device_eval_batch_size=16,
        num_train_epochs=2,

        predict_with_generate=False,
        push_to_hub=False,
        logging_dir="../models/logs",
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()