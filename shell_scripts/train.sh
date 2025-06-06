#!/bin/bash

export CARDINAL_DATA_PATH=$1
export OUTPUT_DIR=$2
export COMET_PROJECT_NAME=$3

# tiny ft-transformer (random weights init)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=tiny-ft-transformer-random trainer.enable_progress_bar=False task/data=tab-clinical-only,tab-clinical-only+ts,tab-all,tab-all+ts 'split_idx=range(5)' >>$OUTPUT_DIR/tiny-ft-transformer-rand.log 2>&1

# tiny ft-transformer (pretraining)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=tiny-ft-transformer-pretrain trainer.enable_progress_bar=False task/data=tab-clinical-only,tab-clinical-only+ts,tab-all,tab-all+ts 'split_idx=range(5)' >>$OUTPUT_DIR/tiny-ft-transformer-pretrain.log 2>&1

# tiny ft-transformer (pretrained weights init)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=tiny-ft-transformer-finetune trainer.enable_progress_bar=False task/data=tab-clinical-only,tab-clinical-only+ts,tab-all,tab-all+ts 'split_idx=range(5)' 'ckpt="${oc.env:OUTPUT_DIR}/tiny-ft-transformer-pretrain/data=${hydra:runtime.choices.task/data}/contrastive=1/encoder=${hydra:runtime.choices.task/model/encoder},cross_blocks=${task.model.encoder.n_bidirectional_blocks}/time_series_tokenizer=${hydra:runtime.choices.task/time_series_tokenizer/model}/cls_token=${task.cls_token},seq_pool=${task.sequence_pooling}/ordinal_mode=True/split_idx=${split_idx}/cardinal_default.ckpt"' >>$OUTPUT_DIR/tiny-ft-transformer-finetune,data=tab+ts.log 2>&1

# xtab
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-clinical-only,tab-clinical-only+ts,tab-all,tab-all+ts ckpt=null,$HOME/data/models/xtab/iter_2k_patch.ckpt 'split_idx=range(5)' >>$OUTPUT_DIR/xtab.log 2>&1

# xtab (ablation study of time-series tokenization and bidirectional attention)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-clinical-only+ts task/time_series_tokenizer/model=transformer task.model.encoder.n_bidirectional_blocks=2 'split_idx=range(5)' >>$OUTPUT_DIR/xtab_multimodal_ablation_study.log 2>&1
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-clinical-only+ts task/time_series_tokenizer/model=linear-embedding task.model.encoder.n_bidirectional_blocks=0,2 'split_idx=range(5)' >>$OUTPUT_DIR/xtab_multimodal_ablation_study.log 2>&1

# xtab (ablation study of output token representation and ordinal mode)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-clinical-only+ts task.ordinal_mode=False 'split_idx=range(5)' >>$OUTPUT_DIR/xtab_latent_representation_ablation_study.log 2>&1
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-clinical-only+ts task.cls_token=False task.sequence_pooling=True task.ordinal_mode=False,True 'split_idx=range(5)' >>$OUTPUT_DIR/xtab_latent_representation_ablation_study.log 2>&1
