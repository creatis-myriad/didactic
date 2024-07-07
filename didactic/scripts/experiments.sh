# xtab-finetune
# tabular data only
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=xtab-finetune trainer.enable_progress_bar=False task/data=tab-selec,tab-no-echo-data,tab-all task.contrastive_loss_weight=0,0.2 task.cls_token=False,True task.ordinal_mode=False,True ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tabular-only.log 2>&1
# tabular + time-series data
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=8 +experiment=xtab-finetune trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task.contrastive_loss_weight=0,0.2 task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True task.ordinal_mode=False,True ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tabular+ts.log 2>&1

# bidirectional-xtab
# tabular + time-series data
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-bidirectional-xtab didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=8 +experiment=bidirectional-xtab trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task.contrastive_loss_weight=0,0.2 task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True task.ordinal_mode=False,True ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt '+trial=range(10)' >>$HOME/data/didactic/results/bidirectional-xtab,data=tabular+ts.log 2>&1

# random
# tabular data only
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-random didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=random trainer.enable_progress_bar=False task/data=tab-selec,tab-no-echo-data,tab-all task.contrastive_loss_weight=0,0.2 task.cls_token=False,True task.ordinal_mode=False,True '+trial=range(10)' >>$HOME/data/didactic/results/random,data=tabular-only.log 2>&1
# tabular + time-series data
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-random didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=random trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task.contrastive_loss_weight=0,0.2 task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True task.ordinal_mode=False,True '+trial=range(10)' >>$HOME/data/didactic/results/random,data=tabular+ts.log 2>&1

# pretrain
# tabular data only
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=pretrain trainer.enable_progress_bar=False task/data=tab-selec,tab-no-echo-data,tab-all task.cls_token=False,True '+trial=range(10)' >>$HOME/data/didactic/results/pretrain,data=tabular-only.log 2>&1
# tabular + time-series data
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=pretrain trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True '+trial=range(10)' >>$HOME/data/didactic/results/pretrain,data=tabular+ts.log 2>&1

# finetune
# tabular data only
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=finetune trainer.enable_progress_bar=False task/data=tab-selec,tab-no-echo-data,tab-all task.contrastive_loss_weight=0,0.2 task.cls_token=False,True task.ordinal_mode=False,True '+trial=range(10)' 'ckpt="/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation/pretrain/data=${hydra:runtime.choices.task/data}/encoder=cardinal-ft-transformer,cross_blocks=0,time_series_tokenizer=${hydra:runtime.choices.task/time_series_tokenizer/model}/cls_token=${task.cls_token}/+trial=${trial}/cardinal_default.ckpt"' >>$HOME/data/didactic/results/finetune,data=tabular-only.log 2>&1
# tabular + time-series data
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=16 +experiment=finetune trainer.enable_progress_bar=False task/data=tab-selec+ts,tab-no-echo-data+ts,tab-all+ts task.contrastive_loss_weight=0,0.2 task/time_series_tokenizer/model=linear-embedding,transformer task.cls_token=False,True task.ordinal_mode=False,True '+trial=range(10)' 'ckpt="/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation/pretrain/data=${hydra:runtime.choices.task/data}/encoder=cardinal-ft-transformer,cross_blocks=0,time_series_tokenizer=${hydra:runtime.choices.task/time_series_tokenizer/model}/cls_token=${task.cls_token}/+trial=${trial}/cardinal_default.ckpt"' >>$HOME/data/didactic/results/finetune,data=tabular+ts.log 2>&1

# records-xgb
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-records-stratification python ~/remote/didactic/didactic/tasks/cardiac_records_stratification.py -m hydra/launcher=joblib hydra.launcher.n_jobs=10 task/data=tab-selec,tab-no-echo-data,tab-all ~task.time_series_attrs task.target_attr=ht_severity '+trial=range(10)' >>$HOME/data/didactic/results/records-xgb.log 2>&1


# Map some configuration options (i.e. data, task) to their respective configuration choices
declare -A tasks_data
tasks_data=(
  [random]="tab-selec tab-selec+ts tab-no-echo-data tab-no-echo-data+ts tab-all tab-all+ts"
  [finetune]="tab-selec tab-selec+ts tab-no-echo-data tab-no-echo-data+ts tab-all tab-all+ts"
  [xtab-finetune]="tab-selec tab-selec+ts tab-no-echo-data tab-no-echo-data+ts tab-all tab-all+ts"
  [bidirectional-xtab]="tab-selec+ts tab-no-echo-data+ts tab-all+ts"
)
declare -A time_series_tokenizers
time_series_tokenizers=(
  [tab-selec]=None
  [tab-selec+ts]="linear-embedding transformer"
  [tab-no-echo-data]=None
  [tab-no-echo-data+ts]="linear-embedding transformer"
  [tab-all]=None
  [tab-all+ts]="linear-embedding transformer"
)
declare -A cross_blocks
cross_blocks=(
  [random]="0"
  [finetune]="0"
  [xtab-finetune]="0"
  [bidirectional-xtab]="2"
)

# Configure the paths for the source and target directories
src_path=$HOME/data/didactic/results/multirun/cardiac-multimodal-representation
target_path=$HOME/data/didactic/results/compiled_results

# Compile the prediction scores over the different trials of each config
for task in "${!tasks_data[@]}"; do
  for data in ${tasks_data[${task}]}; do
    for contrastive in 0 0.2; do
      for cross_blocks in ${cross_blocks[${task}]}; do
        for time_series_tokenizer in ${time_series_tokenizers[${data}]}; do
          for cls_token in False True; do

            for ordinal_mode in False True; do
              run_path="$task/data=$data/contrastive=$contrastive/cross_blocks=$cross_blocks,time_series_tokenizer=$time_series_tokenizer/cls_token=$cls_token/ordinal_mode=$ordinal_mode"

              # Copy model checkpoints to target directory
              python ~/remote/didactic/didactic/scripts/copy_model_ckpt.py $(find $src_path/$run_path -maxdepth 2 -name *.ckpt | sort | tr "\n" " ") --copy_filename='{}.ckpt' --output_dir=$target_path/$run_path >>$target_path/copy_model_ckpt.log 2>&1

              # Compile the prediction scores over the different trials of each config
              echo "Compiling predictions scores for $run_path model" >>$target_path/agg_prediction_scores.log 2>&1
              for scores in train_categorical_scores val_categorical_scores test_categorical_scores; do
                python ~/remote/didactic/didactic/scripts/compile_prediction_scores.py $(find $src_path/$run_path -name $scores.csv | sort | tr "\n" " ") --output_file=$target_path/$run_path/$scores.csv
              done

              # Split patients into bins w.r.t. continuum param predicted for each patient
              for model_id in $(seq 0 9); do
                echo "Splitting patients into bins w.r.t. continuum param for $run_path/$model_id model" >>$target_path/group_patients_by_continuum_param.log 2>&1
                python ~/remote/didactic/didactic/scripts/group_patients_by_predictions.py $target_path/$run_path/$model_id.ckpt --data_roots $HOME/dataset/cardinal/v1.1/data --views A4C A2C --bins=6 --output_dir=$target_path/$run_path/$model_id/continuum_param_bins range --bounds 0 1 >>$target_path/group_patients_by_continuum_param.log 2>&1
              done

              # Plot 2D embeddings of the transformer encoder representations
              for model_id in $(seq 0 9); do
                echo "Generating 2D embedding of latent space for $run_path/$model_id model, using PaCMAP" >>$target_path/2d_embeddings.log 2>&1
                python ~/remote/didactic/didactic/scripts/cardiac_multimodal_representation_plot.py $target_path/$run_path/$model_id.ckpt --data_roots $HOME/dataset/cardinal/v1.1/data --views A4C A2C --plot_categorical_attrs_dirs $HOME/dataset/cardinal/v1.1/patients_by_attr_label/subset '--num_plot_kwargs={palette:flare}' --output_dir=$target_path/$run_path/$model_id/2d_embeddings >>$target_path/2d_embeddings.log 2>&1
              done

            done

            # Compile results specific to ordinal models
            ordinal_mode=True
            run_path="$task/data=$data/contrastive=$contrastive/cross_blocks=$cross_blocks,time_series_tokenizer=$time_series_tokenizer/cls_token=$cls_token/ordinal_mode=$ordinal_mode"

            # Compute the alignment scores between the different trials of each config
            # NOTE: Skip alignment scores for non-ordinal models since the scores rely on the ordinal predictions
            echo "Computing alignment scores between $run_path models" >>$target_path/score_models_alignment.log 2>&1
            python ~/remote/didactic/didactic/scripts/score_models_alignment.py $(find $target_path/$run_path -name *.ckpt | sort | tr "\n" " ") --data_roots $HOME/dataset/cardinal/v1.1/data --views A4C A2C --output_file=$target_path/$run_path/alignment_scores.csv >>$target_path/score_models_alignment.log 2>&1

            # Plot 2D embeddings of the transformer encoder representations w.r.t. continuum bins
            for model_id in $(seq 0 9); do
              echo "Generating 2D embedding of latent space w.r.t. continuum bins for $run_path/$model_id model, using PaCMAP" >>$target_path/2d_embeddings_continuum.log 2>&1
              python ~/remote/didactic/didactic/scripts/cardiac_multimodal_representation_plot.py $target_path/$run_path/$model_id.ckpt --data_roots $HOME/dataset/cardinal/v1.1/data --views A4C A2C --plot_tabular_attrs --plot_categorical_attrs_dirs $HOME/dataset/cardinal/v1.1/patients_by_attr_label/subset '--num_plot_kwargs={palette:flare}' --output_dir=$target_path/$run_path/$model_id/2d_embeddings >>$target_path/2d_embeddings_continuum.log 2>&1
            done

            # Plot variability of attributes w.r.t. predicted continuum
            for model_id in $(seq 0 9); do
              echo "Plotting variability of attrs w.r.t. continuum predicted by $run_path/$model_id model" >>$target_path/attrs_wrt_continuum_bins.log 2>&1
              python ~/remote/didactic/vital/vital/data/cardinal/plot_attrs_wrt_groups.py --data_roots $HOME/dataset/cardinal/v1.1/data --views A4C A2C --groups_txt $(find $target_path/$run_path/$model_id/continuum_param_bins -name "*.txt" | sort | tr "\n" " ") '--time_series_plot_kwargs={errorbar:,palette:flare}' --output_dir=$target_path/$run_path/$model_id/attrs_wrt_continuum_param >>$HOME/data/didactic/results/attrs_wrt_continuum_param.log 2>&1
            done

            # Plot variability of predicted continuum w.r.t. position along the continuum
            echo "Plotting variability of predicted continuum w.r.t. position along the continuum between $run_path models" >>$target_path/plot_models_variability.log 2>&1
            python ~/remote/didactic/didactic/scripts/plot_models_variability.py $(find $target_path/$run_path -name *.ckpt | sort | tr "\n" " ") --data_roots $HOME/dataset/cardinal/v1.1/data --views A4C A2C --hue_attr=ht_severity --output_dir=$target_path/$run_path/continuum_param_variability >>$target_path/plot_models_variability.log 2>&1

          done
        done
      done
    done
  done
done

# Compile the records-xgb prediction scores over the different trials of each config
for task in records-xgb; do
  for data in tab-selec tab-no-echo-data tab-all; do
    for target in ht_severity; do
      src_path="$task/data=$data/target=$target"
      target_path="$task/$data/$target"
      for scores in test_categorical_scores; do
        python ~/remote/didactic/didactic/scripts/compile_prediction_scores.py $(find ~/data/didactic/results/multirun/cardiac-multimodal-representation/$src_path -name $scores.csv | sort | tr "\n" " ") --output_file=$HOME/data/didactic/results/cardiac-multimodal-representation/$target_path/$scores.csv
      done
    done
  done
done
