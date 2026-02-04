import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from models import ModelWrapper
from utils import DataLoaderWrapper, TrainerWrapper
from utils import load_config, get_default_config_path, save_config


def parse_args():
    parser = argparse.ArgumentParser(description='Skin cancer detection pipeline')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-e', '--exp_name', type=str, default=None, help='Experiment name (overrides config)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = args.config if args.config else get_default_config_path()
    config = load_config(config_path)

    # Override experiment name
    config['experiment']['name'] = args.exp_name if args.exp_name else config['experiment']['name']

    # Extract config sections
    exp_config = config['experiment']
    data_config = config['data']
    trainer_config = config['trainer']
    metrics_config = config['metrics']
    metrics_config['num_classes'] = data_config['num_classes']
    model_config = config['model']
    model_config['num_labels'] = data_config['num_classes']
    model_config['_metrics_config'] = metrics_config

    # Output directory
    output_dir = os.path.join(exp_config['base_output_dir'], exp_config['name'])
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, os.path.join(output_dir, 'config.yaml'))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=exp_config['save_top_k'],
        mode='min',
        every_n_epochs=exp_config['check_val_every_n_epoch'],
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        min_delta=1e-4,
        verbose=True,
    )

    # Loggers
    csv_logger = CSVLogger(save_dir=output_dir, name='metrics_logs')
    tensorboard_logger = TensorBoardLogger(save_dir=output_dir, name='tensorboard_logs')

    # Trainer config
    trainer_config.update({
        'check_val_every_n_epoch': exp_config['check_val_every_n_epoch'],
        'callbacks': [checkpoint_callback, early_stopping_callback],
        'logger': [csv_logger, tensorboard_logger],
    })

    # Data & model
    data_wrapper = DataLoaderWrapper(**data_config)
    data_wrapper.setup()

    model_wrapper = ModelWrapper(**model_config)
    checkpoint_path = config.get('resume', {}).get('from_checkpoint', None)
    trainer = TrainerWrapper(checkpoint=checkpoint_path, config=config, **trainer_config)

    # Train
    trainer.fit(model_wrapper, data_wrapper.train_dataloader(), data_wrapper.val_dataloader())

    # Evaluate with thresholds
    print("\nEvaluating model with multiple thresholds...")
    best_th, all_results = trainer.evaluate_with_thresholds(model_wrapper, data_wrapper.val_dataloader())

    for th, metrics in all_results.items():
        print(f"Threshold {th:.2f}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

    # Save best threshold
    with open(os.path.join(output_dir, 'best_threshold.txt'), 'w') as f:
        f.write(f"Best threshold: {best_th} (based on {config['metrics'].get('metric_priority', 'f1')}: "
                f"{all_results[best_th][config['metrics'].get('metric_priority', 'f1')]:.4f})")


if __name__ == "__main__":
    main()
