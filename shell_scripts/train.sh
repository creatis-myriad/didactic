#!/bin/bash

export CARDIAC_MULTIMODAL_REPR_PATH=$HOME/data/didactic/results/5-fold-cv/multirun
export COMET_PROJECT_NAME=didactic-5-fold-cv

# xtab
# tabular data only
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-selec,tab-no-echo-data,tab-all task.cls_token=False,True task.ordinal_mode=False,True ckpt=null,$HOME/data/models/xtab/iter_2k_patch.ckpt 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/xtab,data=tab.log 2>&1
# tabular + time-series data
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=8 +experiment=xtab trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True task.ordinal_mode=False,True ckpt=null,$HOME/data/models/xtab/iter_2k_patch.ckpt 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/xtab,data=tab+ts.log 2>&1

# bidirectional-xtab
# tabular + time-series data
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=8 +experiment=xtab task.model.encoder.n_bidirectional_blocks=2 trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True task.ordinal_mode=False,True ckpt=null,$HOME/data/models/xtab/iter_2k_inc-blocks+2.ckpt 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/bidirectional-xtab,data=tab+ts.log 2>&1

# tiny ft-transformer (random weights init)
# tabular data only
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=tiny-ft-transformer-random trainer.enable_progress_bar=False task/data=tab-selec,tab-no-echo-data,tab-all task.cls_token=False,True task.ordinal_mode=False,True 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-rand,data=tab.log 2>&1
# tabular + time-series data
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=tiny-ft-transformer-random trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True task.ordinal_mode=False,True 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-rand,data=tab+ts.log 2>&1

# tiny ft-transformer (pretraining)
# tabular data only
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=tiny-ft-transformer-pretrain trainer.enable_progress_bar=False task/data=tab-selec,tab-no-echo-data,tab-all task.cls_token=False,True 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-pretrain,data=tab.log 2>&1
# tabular + time-series data
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=tiny-ft-transformer-pretrain trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True 'split_idx=range(5)' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-pretrain,data=tab+ts.log 2>&1

# tiny ft-transformer (pretrained weights init)
# tabular data only
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=tiny-ft-transformer-finetune trainer.enable_progress_bar=False task/data=tab-selec,tab-no-echo-data,tab-all task.cls_token=False,True task.ordinal_mode=False,True 'split_idx=range(5)' 'ckpt="${oc.env:CARDIAC_MULTIMODAL_REPR_PATH}/tiny-ft-transformer-pretrain/data=${hydra:runtime.choices.task/data}/encoder=${hydra:runtime.choices.task/model/encoder},cross_blocks=${task.model.encoder.n_bidirectional_blocks},time_series_tokenizer=${hydra:runtime.choices.task/time_series_tokenizer/model}/cls_token=${task.cls_token}/ordinal_mode=True/split_idx=${split_idx}/cardinal_default.ckpt"' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-finetune,data=tab.log 2>&1
# tabular + time-series data
didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=tiny-ft-transformer-finetune trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True task.ordinal_mode=False,True 'split_idx=range(5)' 'ckpt="${oc.env:CARDIAC_MULTIMODAL_REPR_PATH}/tiny-ft-transformer-pretrain/data=${hydra:runtime.choices.task/data}/encoder=${hydra:runtime.choices.task/model/encoder},cross_blocks=${task.model.encoder.n_bidirectional_blocks},time_series_tokenizer=${hydra:runtime.choices.task/time_series_tokenizer/model}/cls_token=${task.cls_token}/ordinal_mode=True/split_idx=${split_idx}/cardinal_default.ckpt"' >>$CARDIAC_MULTIMODAL_REPR_PATH/tiny-ft-transformer-finetune,data=tab+ts.log 2>&1
