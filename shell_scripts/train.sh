#!/bin/bash

export CARDIAC_MULTIMODAL_REPR_PATH=$HOME/data/didactic/results/5-fold-cv/multirun
export COMET_PROJECT_NAME=didactic-5-fold-cv

# tiny ft-transformer (random weights init)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=tiny-ft-transformer-random trainer.enable_progress_bar=False task/data=tab-no-echo-data,tab-no-echo-data+ts,tab-all,tab-all+ts 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-rand.log 2>&1

# tiny ft-transformer (pretraining)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=tiny-ft-transformer-pretrain trainer.enable_progress_bar=False task/data=tab-no-echo-data,tab-no-echo-data+ts,tab-all,tab-all+ts 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-pretrain.log 2>&1

# tiny ft-transformer (pretrained weights init)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=tiny-ft-transformer-finetune trainer.enable_progress_bar=False task/data=tab-no-echo-data,tab-no-echo-data+ts,tab-all,tab-all+ts 'split_idx=range(5)' 'ckpt="${oc.env:CARDIAC_MULTIMODAL_REPR_PATH}/tiny-ft-transformer-pretrain/data=${hydra:runtime.choices.task/data}/contrastive=1/encoder=${hydra:runtime.choices.task/model/encoder},cross_blocks=${task.model.encoder.n_bidirectional_blocks}/time_series_tokenizer=${hydra:runtime.choices.task/time_series_tokenizer/model}/cls_token=${task.cls_token}/ordinal_mode=True/split_idx=${split_idx}/cardinal_default.ckpt"' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-finetune,data=tab+ts.log 2>&1

# xtab
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-no-echo-data,tab-no-echo-data+ts,tab-all,tab-all+ts ckpt=null,$HOME/data/models/xtab/iter_2k_patch.ckpt 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/xtab.log 2>&1

# xtab (ablation study of time-series tokenization and bidirectional attention)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=8 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-no-echo-data+ts task/time_series_tokenizer/model=linear-embedding,transformer task.model.encoder.n_bidirectional_blocks=0,2 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/xtab_multimodal_ablation_study.log 2>&1

# xtab (ablation study of output token representation and ordinal mode)
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=8 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-no-echo-data+ts task.cls_token=False,True task.ordinal_mode=False,True 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/xtab_latent_representation_ablation_study.log 2>&1
