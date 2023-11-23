# xtab-finetune
# w/ time-series data + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task/data=tab-13+time-series,tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tab-13+ts,tab+ts,ordinal=False.log 2>&1
# w/ time-series data + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task/data=tab-13+time-series,tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tab-13+ts,tab+ts,ordinal=True.log 2>&1
# w/o time-series data + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task/data=tab-13,records,tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tab-13,tab,ordinal=False.log 2>&1
# w/o time-series data + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task/data=tab-13,records,tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tab-13,tab,ordinal=True.log 2>&1

# scratch
# w/ time-series data + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-scratch trainer.enable_progress_bar=False task/data=tab-13+time-series,tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-scratch,data=tab-13+ts,tab+ts,ordinal=False.log 2>&1
# w/ time-series data + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-scratch trainer.enable_progress_bar=False task/data=tab-13+time-series,tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-scratch,data=tab-13+ts,tab+ts,ordinal=True.log 2>&1
# w/o time-series data + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-scratch trainer.enable_progress_bar=False task/data=tab-13,records,tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-scratch,data=tab-13,tab,ordinal=False.log 2>&1
# w/o time-series data + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-scratch trainer.enable_progress_bar=False task/data=tab-13,records,tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-scratch,data=tab-13,tab,ordinal=True.log 2>&1

# pretrain
# tab-13
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=tab-13 '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=tab-13.log 2>&1
# tab-13+time-series
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=tab-13+time-series task/time_series_tokenizer/model=linear-embedding,transformer '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=tab-13+ts.log 2>&1
# records
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=records exclude_tabular_attrs=[ht_severity,ht_grade] '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=records.log 2>&1
# tabular
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=tabular exclude_tabular_attrs=[ht_severity,ht_grade] '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=tab.log 2>&1
# tabular+time-series
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=tabular+time-series task/time_series_tokenizer/model=linear-embedding,transformer exclude_tabular_attrs=[ht_severity,ht_grade] '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=tab+ts.log 2>&1

# finetune
# tab-13 + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=tab-13 task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/None/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=tab-13,ordinal=False.log 2>&1
# tab-13 + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=tab-13 task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/None/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=tab-13,ordinal=True.log 2>&1
# tab-13+time-series + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=tab-13+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/${hydra:runtime.choices.task/time_series_tokenizer/model}/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=tab-13+ts,ordinal=False.log 2>&1
# tab-13+time-series + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=tab-13+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/${hydra:runtime.choices.task/time_series_tokenizer/model}/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=tab-13+ts,ordinal=True.log 2>&1
# records + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=records task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/None/ht_severity/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=records,ordinal=False.log 2>&1
# records + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=records task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/None/ht_severity/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=records,ordinal=True.log 2>&1
# tabular + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/None/ht_severity/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=tab,ordinal=False.log 2>&1
# tabular + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/None/ht_severity/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=tab,ordinal=True.log 2>&1
# tabular+time-series + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/${hydra:runtime.choices.task/time_series_tokenizer/model}/ht_severity/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=tab+ts,ordinal=False.log 2>&1
# tabular+time-series + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-finetune trainer.enable_progress_bar=False task/data=tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/results/cardiac-multimodal-representation/pretrain/${hydra:runtime.choices.task/data}/${hydra:runtime.choices.task/time_series_tokenizer/model}/ht_severity/${trial}.ckpt' >>$HOME/data/didactic/results/multimodal-xformer-finetune,data=tab+ts,ordinal=True.log 2>&1


# Map the time-series tokenizers available for each data option
declare -A time_series_tokenizers
time_series_tokenizers=(
  [tab-13]=None
  [tab-13+time-series]="linear-embedding transformer"
  [records]=None
  [tabular]=None
  [tabular+time-series]="linear-embedding transformer"
)

# Compile the prediction scores over the different trials of each config
for task in scratch finetune xtab-finetune; do
  for contrastive in 0 0.2 1; do
    for data in "${!time_series_tokenizers[@]}"; do
      for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
        for target in ht_severity; do
          # begin w/o ordinal constraint
          ordinal_mode=False
          src_path="$task/data=$data/contrastive=$contrastive/time_series_tokenizer=$time_series_tokenizer/targets=['$target']/ordinal_mode=$ordinal_mode,distribution=binomial,tau_mode=learn_fn"
          target_path="$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode"
          for scores in test_categorical_scores; do
            python ~/remote/didactic/didactic/scripts/compile_prediction_scores.py $(find ~/data/didactic/results/multirun/cardiac-multimodal-representation/$src_path -name $scores.csv | sort | tr "\n" " ") --output_file=$HOME/data/didactic/results/cardiac-multimodal-representation/$target_path/$scores.csv
          done
          # end w/o ordinal constraint
          # begin w/ ordinal constraint
          ordinal_mode=True
          for distribution in poisson binomial; do
            for tau_mode in learn_sigm learn_fn; do
              src_path="$task/data=$data/contrastive=$contrastive/time_series_tokenizer=$time_series_tokenizer/targets=['$target']/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode"
              target_path="$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode"
              for scores in test_categorical_scores; do
                python ~/remote/didactic/didactic/scripts/compile_prediction_scores.py $(find ~/data/didactic/results/multirun/cardiac-multimodal-representation/$src_path -name $scores.csv | sort | tr "\n" " ") --output_file=$HOME/data/didactic/results/cardiac-multimodal-representation/$target_path/$scores.csv
              done
            done
          done
          # end w/ ordinal constraint
        done
      done
    done
  done
done


# Copy the model checkpoints
rm $HOME/data/didactic/results/copy_model_ckpt.log
for task in scratch finetune xtab-finetune; do
  for contrastive in 0 0.2 1; do
    for data in "${!time_series_tokenizers[@]}"; do
      for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
        for target in ht_severity; do
          # begin w/o ordinal constraint
          ordinal_mode=False
          src_path="$task/data=$data/contrastive=$contrastive/time_series_tokenizer=$time_series_tokenizer/targets=['$target']/ordinal_mode=$ordinal_mode,distribution=binomial,tau_mode=learn_fn"
          target_path="$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode"
          python ~/remote/didactic/didactic/scripts/copy_model_ckpt.py $(find ~/data/didactic/results/multirun/cardiac-multimodal-representation/$src_path -maxdepth 2 -name *.ckpt | sort | tr "\n" " ") --copy_filename='{}.ckpt' --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$target_path >>$HOME/data/didactic/results/copy_model_ckpt.log 2>&1
          # end w/o ordinal constraint
          # begin w/ ordinal constraint
          ordinal_mode=True
          for distribution in poisson binomial; do
            for tau_mode in learn_sigm learn_fn; do
              src_path="$task/data=$data/contrastive=$contrastive/time_series_tokenizer=$time_series_tokenizer/targets=['$target']/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode"
              target_path="$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode"
              python ~/remote/didactic/didactic/scripts/copy_model_ckpt.py $(find ~/data/didactic/results/multirun/cardiac-multimodal-representation/$src_path -maxdepth 2 -name *.ckpt | sort | tr "\n" " ") --copy_filename='{}.ckpt' --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$target_path >>$HOME/data/didactic/results/copy_model_ckpt.log 2>&1
            done
          done
          # end w/ ordinal constraint
        done
      done
    done
  done
done

# Copy the model checkpoints (self-supervised pretraining)
declare -A exclude_tabular_attrs
exclude_tabular_attrs=(
  [ht_severity]="['ht_severity', 'ht_grade']"
)
rm $HOME/data/didactic/results/copy_pretrain_model_ckpt.log
for task in pretrain; do
  # begin w/ tab-13 data (no tabular attrs excluded in this case)
  for data in tab-13 tab-13+time-series; do
    for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
      src_path="$task/data=$data/contrastive=$contrastive/time_series_tokenizer=$time_series_tokenizer/exclude_tabular_attrs=[]"
      target_path="$task/$data/$time_series_tokenizer"
      python ~/remote/didactic/didactic/scripts/copy_model_ckpt.py $(find ~/data/didactic/results/multirun/cardiac-multimodal-representation/$src_path -maxdepth 2 -name *.ckpt | sort | tr "\n" " ") --copy_filename='{}.ckpt' --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$target_path >>$HOME/data/didactic/results/copy_pretrain_model_ckpt.log 2>&1
    done
  done
  # end w/ tab-13 data
  # begin w/ tabular data
  # TOFIX Escaping of space in paths returned by `find` to get the model ckpt paths is not working in script
  #       (but similar command works fine interactively)
  for data in records tabular tabular+time-series; do
    for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
      for target in "${!exclude_tabular_attrs[@]}"; do
        src_path="$task/data=$data/contrastive=$contrastive/time_series_tokenizer=$time_series_tokenizer/exclude_tabular_attrs=${exclude_tabular_attrs[${target}]}"
        target_path="$task/$data/$time_series_tokenizer/$target"
        python ~/remote/didactic/didactic/scripts/copy_model_ckpt.py $(find "$HOME/data/didactic/results/multirun/cardiac-multimodal-representation/$src_path" -maxdepth 2 -name *.ckpt -printf '"%p"\n' | sort | sed "s/'/'\\\''/g" | tr '"' "'" | tr "\n" " ") --copy_filename='{}.ckpt' --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$target_path >>$HOME/data/didactic/results/copy_pretrain_model_ckpt.log 2>&1
      done
    done
  done
  # end w/ tabular data
done


# Split patients into bins w.r.t. unimodal param predicted for each patient
rm $HOME/data/didactic/results/group_patients_by_unimodal_param.log
for task in scratch finetune xtab-finetune; do
  for contrastive in 0 0.2 1; do
    for data in "${!time_series_tokenizers[@]}"; do
      for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
        for target in ht_severity; do
          # Skip patients binning for non-ordinal models since the bins rely on the ordinal head predictions
          # begin w/ ordinal constraint
          ordinal_mode=True
          for distribution in poisson binomial; do
            for tau_mode in learn_sigm learn_fn; do
              job_path=$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode
              for model_id in $(seq 0 9); do
                echo "Splitting patients into bins w.r.t. unimodal param for $job_path/$model_id model" >>$HOME/data/didactic/results/group_patients_by_unimodal_param.log 2>&1
                python ~/remote/didactic/didactic/scripts/group_patients_by_predictions.py $HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id.ckpt --data_roots $HOME/dataset/cardinal/v1.0/data --views A4C A2C --bins=8 --bounds 0 1 --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id/unimodal_param_bins >>$HOME/data/didactic/results/group_patients_by_unimodal_param.log 2>&1
              done
            done
          done
          # end w/ ordinal constraint
        done
      done
    done
  done
done


# Compute the alignment scores between the different trials of each config
rm $HOME/data/didactic/results/score_models_alignment.log
for task in scratch finetune xtab-finetune; do
  for contrastive in 0 0.2 1; do
    for data in "${!time_series_tokenizers[@]}"; do
      for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
        for target in ht_severity; do
          # Skip alignment scores for non-ordinal models since the scores rely on the ordinal predictions
          # begin w/ ordinal constraint
          ordinal_mode=True
          for distribution in poisson binomial; do
            for tau_mode in learn_sigm learn_fn; do
              job_path=$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode
              echo "Computing alignment scores between $job_path models" >>$HOME/data/didactic/results/score_models_alignment.log 2>&1
              python ~/remote/didactic/didactic/scripts/score_models_alignment.py $(find ~/data/didactic/results/cardiac-multimodal-representation/$job_path -name *.ckpt | sort | tr "\n" " ") --data_roots $HOME/dataset/cardinal/v1.0/data --views A4C A2C --output_file=$HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/alignment_scores.csv >>$HOME/data/didactic/results/score_models_alignment.log 2>&1
            done
          done
          # end w/ ordinal constraint
        done
      done
    done
  done
done


# Plot 2D embeddings of the transformer encoder representations
rm $HOME/data/didactic/results/2d_embeddings.log
for task in scratch finetune xtab-finetune; do
  for contrastive in 0 0.2 1; do
    for data in "${!time_series_tokenizers[@]}"; do
      for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
        for target in ht_severity; do
          # begin w/o ordinal constraint
          ordinal_mode=False
          job_path=$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode
          for model_id in $(seq 0 9); do
            echo "Generating 2D embedding of latent space of $job_path/$model_id model using PaCMAP" >>$HOME/data/didactic/results/2d_embeddings.log 2>&1
            python ~/remote/didactic/didactic/scripts/cardiac_multimodal_representation_plot.py $HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id.ckpt --data_roots $HOME/dataset/cardinal/v1.0/data --views A4C A2C --plot_categorical_attrs_dirs $HOME/dataset/cardinal/v1.0/patients_by_attr_label/subset '--num_plot_kwargs={palette:flare}' --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id/2d_embeddings >>$HOME/data/didactic/results/2d_embeddings.log 2>&1
          done
          # end w/o ordinal constraint
          # begin w/ ordinal constraint
          ordinal_mode=True
          for distribution in poisson binomial; do
            for tau_mode in learn_sigm learn_fn; do
              job_path=$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode
              for model_id in $(seq 0 9); do
                echo "Generating 2D embedding of latent space of $job_path/$model_id model using PaCMAP" >>$HOME/data/didactic/results/2d_embeddings.log 2>&1
                python ~/remote/didactic/didactic/scripts/cardiac_multimodal_representation_plot.py $HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id.ckpt --data_roots $HOME/dataset/cardinal/v1.0/data --views A4C A2C --plot_categorical_attrs_dirs $HOME/dataset/cardinal/v1.0/patients_by_attr_label/subset $HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id/unimodal_param_bins '--num_plot_kwargs={palette:flare}' --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id/2d_embeddings >>$HOME/data/didactic/results/2d_embeddings.log 2>&1
              done
            done
          done
          # end w/ ordinal constraint
        done
      done
    done
  done
done


# Plot variability of attributes w.r.t. predicted continuum
rm $HOME/data/didactic/results/attrs_wrt_unimodal_param.log
for task in scratch finetune xtab-finetune; do
  for contrastive in 0 0.2 1; do
    for data in "${!time_series_tokenizers[@]}"; do
      for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
        for target in ht_severity; do
          # Skip attributes w.r.t. predicted continuum for non-ordinal models since the continuum relies on the ordinal predictions
          # begin w/ ordinal constraint
          ordinal_mode=True
          for distribution in poisson binomial; do
            for tau_mode in learn_sigm learn_fn; do
              job_path=$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode
              for model_id in $(seq 0 9); do
                echo "Plotting variability of attrs w.r.t. continuum predicted by $job_path/$model_id model" >>$HOME/data/didactic/results/attrs_wrt_unimodal_param.log 2>&1
                python ~/remote/didactic/vital/vital/data/cardinal/plot_attrs_wrt_groups.py --data_roots $HOME/dataset/cardinal/v1.0/data --views A4C A2C --groups_txt $(find ~/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id/unimodal_param_bins -name "*.txt" | sort | tr "\n" " ") '--time_series_plot_kwargs={errorbar:,palette:flare}' --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/$model_id/attrs_wrt_unimodal_param >>$HOME/data/didactic/results/attrs_wrt_unimodal_param.log 2>&1
              done
            done
          done
          # end w/ ordinal constraint
        done
      done
    done
  done
done


# Plot variability of predicted continuum w.r.t. position along the continuum
rm $HOME/data/didactic/results/plot_models_variability.log
for task in scratch finetune xtab-finetune; do
  for contrastive in 0 0.2 1; do
    for data in "${!time_series_tokenizers[@]}"; do
      for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
        for target in ht_severity; do
          # Skip attributes w.r.t. predicted continuum for non-ordinal models since the continuum relies on the ordinal predictions
          # begin w/ ordinal constraint
          ordinal_mode=True
          for distribution in poisson binomial; do
            for tau_mode in learn_sigm learn_fn; do
              job_path=$task/contrastive=$contrastive/$data/$time_series_tokenizer/$target/ordinal_mode=$ordinal_mode,distribution=$distribution,tau_mode=$tau_mode
              echo "Plotting variability of predicted continuum w.r.t. position along the continuum between $job_path models" >>$HOME/data/didactic/results/plot_models_variability.log 2>&1
              python ~/remote/didactic/didactic/scripts/plot_models_variability.py $(find ~/data/didactic/results/cardiac-multimodal-representation/$job_path -name *.ckpt | sort | tr "\n" " ") --data_roots $HOME/dataset/cardinal/v1.0/data --views A4C A2C --hue_attr=ht_severity --output_dir=$HOME/data/didactic/results/cardiac-multimodal-representation/$job_path/unimodal_param_variability >>$HOME/data/didactic/results/plot_models_variability.log 2>&1
            done
          done
          # end w/ ordinal constraint
        done
      done
    done
  done
done
