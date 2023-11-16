# xtab-finetune
# w/ time-series data + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task/data=tab-13+time-series,tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tab-13+ts,tab+ts,ordinal=False.log 2>&1
# w/ time-series data + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=5 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task/data=tab-13+time-series,tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tab-13+ts,tab+ts,ordinal=True.log 2>&1
# w/o time-series data + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task/data=tab-13,tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tab-13,tab,ordinal=False.log 2>&1
# w/o time-series data + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task/data=tab-13,tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)' >>$HOME/data/didactic/results/xtab-finetune,data=tab-13,tab,ordinal=True.log 2>&1

# scratch
# w/ time-series data + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-scratch trainer.enable_progress_bar=False task/data=tab-13+time-series,tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-scratch,data=tab-13+ts,tab+ts,ordinal=False.log 2>&1
# w/ time-series data + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-scratch trainer.enable_progress_bar=False task/data=tab-13+time-series,tabular+time-series task.contrastive_loss_weight=0,0.2,1 task/time_series_tokenizer/model=linear-embedding,transformer 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-scratch,data=tab-13+ts,tab+ts,ordinal=True.log 2>&1
# w/o time-series data + w/o ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-scratch trainer.enable_progress_bar=False task/data=tab-13,tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=False '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-scratch,data=tab-13,tab,ordinal=False.log 2>&1
# w/o time-series data + w/ ordinal constraint
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-scratch trainer.enable_progress_bar=False task/data=tab-13,tabular task.contrastive_loss_weight=0,0.2,1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' exclude_tabular_attrs=[ht_severity,ht_grade] task.ordinal_mode=True task.model.ordinal_head.distribution=poisson,binomial task.model.ordinal_head.tau_mode=learn_sigm,learn_fn '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-scratch,data=tab-13,tab,ordinal=True.log 2>&1

# pretrain
# tab-13
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=tab-13 '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=tab-13.log 2>&1
# tab-13+time-series
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=tab-13+time-series task/time_series_tokenizer/model=linear-embedding,transformer '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=tab-13+ts.log 2>&1
# tabular
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=tabular exclude_tabular_attrs=[ht_severity,ht_grade] '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=tab.log 2>&1
# tabular+time-series
CARDIAC_MULTIMODAL_REPR_PATH=/home/local/USHERBROOKE/pain5474/data/didactic/results/multirun/cardiac-multimodal-representation COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/multimodal-xformer-pretrain trainer.enable_progress_bar=False task/data=tabular+time-series task/time_series_tokenizer/model=linear-embedding,transformer exclude_tabular_attrs=[ht_severity,ht_grade] '+trial=range(10)' >>$HOME/data/didactic/results/multimodal-xformer-pretrain,data=tab+ts.log 2>&1

# finetune
# w/ time-series data + w/o ordinal constraint

# w/ time-series data + w/ ordinal constraint

# w/o time-series data + w/o ordinal constraint

# w/o time-series data + w/ ordinal constraint


# Plot 2D embeddings of the transformer encoder representations
for model in multimodal-xformer
for task in scratch scratch-cotrain finetune finetune-cotrain
for data in tab-13 tab-13+time-series tabular tabular+time-series
for target in ht_severity
set method_path $model/$task/$data
set method_name $model-$task-$data-$target
for model_id in (seq 0 9)
echo "Generating 2D embedding of latent space of $method_name-$model_id model using PaCMAP" >>$HOME/data/didactic/results/2d_embeddings_$model.log 2>&1
python didactic/scripts/cardiac_multimodal_representation_plot.py $HOME/data/didactic/models/$method_path/$method_name-$model_id.ckpt --data_roots $HOME/dataset/cardinal/v1.0/data --views A4C A2C --plot_categorical_attrs_dirs $HOME/dataset/cardinal/v3/patients_by_attr_label/subset --output_dir=$HOME/data/didactic/results/2d_embeddings/$method_path/$target/$model_id >>$HOME/data/didactic/results/2d_embeddings_$model.log 2>&1
end
end
end
end
end

## Cluster the transformer encoder representations
#for model in multimodal-xformer
#for task in scratch scratch-cotrain finetune finetune-cotrain
#for data in tab-13 tab-13+time-series tabular tabular+time-series
#for target in ht_severity
#set method_path $model/$task/$data
#set method_name $model-$task-$data-$target
#for model_id in (seq 0 9)
#for trial in 0
#echo "Running GMM clustering trial #$trial for $method_name-$model_id model"
#python didactic/tasks/cardiac_representation_clustering.py $HOME/data/didactic/models/$method_path/$method_name-$model_id.ckpt --data_roots $HOME/dataset/cardinal/v3/data --views A4C A2C --covariance_type diag --n_components 2 11 --num_sweeps=10 --output_dir=$HOME/data/didactic/results/clustering_hparams_search/$method_path/$target/$model_id/$trial
#end
#end
#end
#end
#end
#end
#
## Running clustering evaluation of transformer encoder representations
#for model in multimodal-xformer
#for task in scratch scratch-cotrain finetune finetune-cotrain
#for data in tab-13 tab-13+time-series tabular tabular+time-series
#for target in ht_severity
#set method_path $model/$task/$data/$target
#set method_name $model-$task-$data-$target
#echo "Evaluating GMM clustering of $method_name models" >>$HOME/data/didactic/results/clustering_eval_$model.log 2>&1
#python didactic/scripts/describe_representation_clustering.py (ls $HOME/data/didactic/results/clustering/$method_path/**/predictions.csv) --data_roots=$HOME/dataset/cardinal/v1.0/data --views A4C A2C --output_dir=$HOME/data/didactic/results/clustering_eval/$method_path >>$HOME/data/didactic/results/clustering_eval_$model.log 2>&1
#end
#end
#end
#end
#
## Running KNN evaluation of transformer encoder representations
#for model in multimodal-xformer
#for task in scratch scratch-cotrain finetune finetune-cotrain
#for data in tab-13 tab-13+time-series tabular tabular+time-series
#for target in ht_severity
#for ref_attr in ht_severity
#set method_path $model/$task/$data
#set method_name $model-$task-$data-$target
#echo "Evaluating KNN representations of $method_name models" >>$HOME/data/didactic/results/knn_eval_$model.log 2>&1
#python didactic/scripts/describe_representation_knn.py (ls $HOME/data/didactic/models/$method_path/*$target*.ckpt) --data_roots $HOME/dataset/cardinal/v1.0/data --views A4C A2C --output_dir=$HOME/data/didactic/results/knn_eval/$method_path/$target '--neigh_kwargs={n_neighbors:8}' '--clinical_plot_kwargs={color:model}' --image_n_bins=8 --reference_attr=$ref_attr >>$HOME/data/didactic/results/knn_eval_$model.log 2>&1
#end
#end
#end
#end
#end
