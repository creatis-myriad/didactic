# xtab-finetune + xtab-finetune-cotrain
## top-13 + top-13+img + all-clin + all
env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.clinical_attrs=[sbp_tte,pp_tte,age,diastolic_dysfunction_param_sum,pw_d,lvm_ind,e_e_prime_ratio,gfr,lateral_e_prime,septal_e_prime,a_velocity,ddd,la_volume],${list.remove:${data.process_patient_kwargs.clinical_attributes},${excluded_clinical_attrs}}' 'task.predict_losses={${list.at:${excluded_clinical_attrs},0}:{_target_:torch.nn.CrossEntropyLoss}}' 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp],[ht_severity,ht_grade] ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'

# separate commands
### top-13 + top-13+img
#env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.clinical_attrs=[sbp_tte,pp_tte,age,diastolic_dysfunction_param_sum,pw_d,lvm_ind,e_e_prime_ratio,gfr,lateral_e_prime,septal_e_prime,a_velocity,ddd,la_volume] task.img_attrs=[],[gls,lv_area,lv_length,myo_area] excluded_clinical_attrs=[] task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}},{ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'
### all-clin + all
#env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={${list.at:${excluded_clinical_attrs},0}:{_target_:torch.nn.CrossEntropyLoss}}' 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp],[ht_severity,ht_grade] ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'
### all-clin + all (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}}' 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp] ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'
### all-clin + all (ht_severity)
#env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[ht_severity,ht_grade] ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'
### all-clin (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}}' task.img_attrs=[] excluded_clinical_attrs=[nt_probnp_group,nt_probnp] ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'
### all-clin (ht_severity)
#env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' task.img_attrs=[] excluded_clinical_attrs=[ht_severity,ht_grade] ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'
### all (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp] ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'
### all (ht_severity)
#env COMET_PROJECT_NAME=didactic-xtab-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/xtab-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' excluded_clinical_attrs=[ht_severity,ht_grade] ckpt=$HOME/data/models/xtab/iter_2k_patch.ckpt ~callbacks.learning_rate_finder '+trial=range(10)'


# scratch + cotrain
## top-13 + top-13+img + all-clin + all
env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.clinical_attrs=[sbp_tte,pp_tte,age,diastolic_dysfunction_param_sum,pw_d,lvm_ind,e_e_prime_ratio,gfr,lateral_e_prime,septal_e_prime,a_velocity,ddd,la_volume],${list.remove:${data.process_patient_kwargs.clinical_attributes},${excluded_clinical_attrs}}' 'task.predict_losses={${list.at:${excluded_clinical_attrs},0}:{_target_:torch.nn.CrossEntropyLoss}}' 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp],[ht_severity,ht_grade] '+trial=range(10)'

# separate commands
### top-13 + top-13+img
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.clinical_attrs=[sbp_tte,pp_tte,age,diastolic_dysfunction_param_sum,pw_d,lvm_ind,e_e_prime_ratio,gfr,lateral_e_prime,septal_e_prime,a_velocity,ddd,la_volume] task.img_attrs=[],[gls,lv_area,lv_length,myo_area] excluded_clinical_attrs=[] task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}},{ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' '+trial=range(10)' ; \
### all-clin + all
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={${list.at:${excluded_clinical_attrs},0}:{_target_:torch.nn.CrossEntropyLoss}}' 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp],[ht_severity,ht_grade] '+trial=range(10)'
### all-clin + all (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}}' 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp] '+trial=range(10)' ; \
### all-clin + all (ht_severity)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[ht_severity,ht_grade] '+trial=range(10)'
### all-clin (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}}' task.img_attrs=[] excluded_clinical_attrs=[nt_probnp_group,nt_probnp] '+trial=range(10)' ; \
### all-clin (ht_severity)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' task.img_attrs=[] excluded_clinical_attrs=[ht_severity,ht_grade] '+trial=range(10)' ; \
### all (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp] '+trial=range(10)' ; \
### all (ht_severity)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-scratch didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-scratch trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' excluded_clinical_attrs=[ht_severity,ht_grade] '+trial=range(10)'


# pretrain
## top-13 + top-13+img
env COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-pretrain trainer.enable_progress_bar=False task.clinical_attrs=[sbp_tte,pp_tte,age,diastolic_dysfunction_param_sum,pw_d,lvm_ind,e_e_prime_ratio,gfr,lateral_e_prime,septal_e_prime,a_velocity,ddd,la_volume] 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[] '+trial=range(10)' ; \
## all-clin + all
env COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-pretrain trainer.enable_progress_bar=False 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp],[ht_severity,ht_grade] '+trial=range(10)'

# separate commands
### all-clin + all (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-pretrain trainer.enable_progress_bar=False 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp] '+trial=range(10)' ; \
### all-clin + all (ht_severity)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-pretrain trainer.enable_progress_bar=False 'task.img_attrs=[],${data.process_patient_kwargs.image_attributes}' excluded_clinical_attrs=[ht_severity,ht_grade] '+trial=range(10)' ; \
### all-clin (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-pretrain trainer.enable_progress_bar=False task.img_attrs=[] excluded_clinical_attrs=[nt_probnp_group,nt_probnp] '+trial=range(10)' ; \
### all-clin (ht_severity)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-pretrain trainer.enable_progress_bar=False task.img_attrs=[] excluded_clinical_attrs=[ht_severity,ht_grade] '+trial=range(10)' ; \
### all (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-pretrain trainer.enable_progress_bar=False excluded_clinical_attrs=[nt_probnp_group,nt_probnp] '+trial=range(10)' ; \
### all (ht_severity)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-pretrain didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-pretrain trainer.enable_progress_bar=False excluded_clinical_attrs=[ht_severity,ht_grade] '+trial=range(10)'


# finetune + finetune-cotrain
## top-13
env COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-finetune trainer.enable_progress_bar=False task.clinical_attrs=[sbp_tte,pp_tte,age,diastolic_dysfunction_param_sum,pw_d,lvm_ind,e_e_prime_ratio,gfr,lateral_e_prime,septal_e_prime,a_velocity,ddd,la_volume] task.img_attrs=[] excluded_clinical_attrs=[] task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}},{ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/models/multimodal-xformer/pretrain/top-13/multimodal-xformer-pretrain-top-13-${trial}.ckpt' '+trial=range(10)' ; \
## top-13+img
env COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-finetune trainer.enable_progress_bar=False task.clinical_attrs=[sbp_tte,pp_tte,age,diastolic_dysfunction_param_sum,pw_d,lvm_ind,e_e_prime_ratio,gfr,lateral_e_prime,septal_e_prime,a_velocity,ddd,la_volume] excluded_clinical_attrs=[] task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}},{ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/models/multimodal-xformer/pretrain/top-13+img/multimodal-xformer-pretrain-top-13+img-${trial}.ckpt' '+trial=range(10)' ; \
## all-clin
env COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={${list.at:${excluded_clinical_attrs},0}:{_target_:torch.nn.CrossEntropyLoss}}' task.img_attrs=[] excluded_clinical_attrs=[nt_probnp_group,nt_probnp],[ht_severity,ht_grade] 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/models/multimodal-xformer/pretrain/all-clin/multimodal-xformer-pretrain-all-clin-${list.at:${excluded_clinical_attrs},0}-${trial}.ckpt' '+trial=range(10)' ; \
## all
env COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={${list.at:${excluded_clinical_attrs},0}:{_target_:torch.nn.CrossEntropyLoss}}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp],[ht_severity,ht_grade] 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/models/multimodal-xformer/pretrain/all/multimodal-xformer-pretrain-all-${list.at:${excluded_clinical_attrs},0}-${trial}.ckpt' '+trial=range(10)'

# separate commands
### all-clin (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}}' task.img_attrs=[] excluded_clinical_attrs=[nt_probnp_group,nt_probnp] 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/models/multimodal-xformer/pretrain/all-clin/multimodal-xformer-pretrain-all-clin-nt_probnp_group-${trial}.ckpt' '+trial=range(10)' ; \
### all-clin (ht_severity)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' task.img_attrs=[] excluded_clinical_attrs=[ht_severity,ht_grade] 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/models/multimodal-xformer/pretrain/all-clin/multimodal-xformer-pretrain-all-clin-ht_severity-${trial}.ckpt' '+trial=range(10)' ; \
### all (nt_probnp_group)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={nt_probnp_group:{_target_:torch.nn.CrossEntropyLoss}}' excluded_clinical_attrs=[nt_probnp_group,nt_probnp] 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/models/multimodal-xformer/pretrain/all/multimodal-xformer-pretrain-all-nt_probnp_group-${trial}.ckpt' '+trial=range(10)' ; \
### all (ht_severity)
#env COMET_PROJECT_NAME=didactic-multimodal-xformer-finetune didactic-runner -m hydra/launcher=joblib hydra.launcher.n_jobs=10 +experiment=cardinal/cardiac-multimodal-representation-finetune trainer.enable_progress_bar=False task.contrastive_loss_weight=0,0.1 'task.predict_losses={ht_severity:{_target_:torch.nn.CrossEntropyLoss}}' excluded_clinical_attrs=[ht_severity,ht_grade] 'ckpt=/home/local/USHERBROOKE/pain5474/data/didactic/models/multimodal-xformer/pretrain/all/multimodal-xformer-pretrain-all-ht_severity-${trial}.ckpt' '+trial=range(10)'


# Plot 2D embeddings of the transformer encoder representations
for model in multimodal-xformer
for task in scratch scratch-cotrain finetune finetune-cotrain
for data in top-13 top-13+img all-clin all
for target in nt_probnp_group ht_severity
set method_path $model/$task/$data
set method_name $model-$task-$data-$target
for model_id in (seq 0 9)
echo "Generating 2D embedding of latent space of $method_name-$model_id model using PaCMAP" >>$HOME/data/didactic/results/2d_embeddings_$model.log 2>&1
python didactic/scripts/cardiac_multimodal_representation_plot.py $HOME/data/didactic/models/$method_path/$method_name-$model_id.ckpt --data_roots $HOME/dataset/cardinal/v4/data --views A4C A2C --plot_categorical_attrs_dirs $HOME/dataset/cardinal/v3/patients_by_attr_label/subset --output_dir=$HOME/data/didactic/results/2d_embeddings/$method_path/$target/$model_id >>$HOME/data/didactic/results/2d_embeddings_$model.log 2>&1
end
end
end
end
end

# Cluster the transformer encoder representations
for model in multimodal-xformer
for task in scratch scratch-cotrain finetune finetune-cotrain
for data in top-13 top-13+img all-clin all
for target in nt_probnp_group ht_severity
set method_path $model/$task/$data
set method_name $model-$task-$data-$target
for model_id in (seq 0 9)
for trial in 0
echo "Running GMM clustering trial #$trial for $method_name-$model_id model"
python didactic/tasks/cardiac_representation_clustering.py $HOME/data/didactic/models/$method_path/$method_name-$model_id.ckpt --data_roots $HOME/dataset/cardinal/v3/data --views A4C A2C --covariance_type diag --n_components 2 11 --num_sweeps=10 --output_dir=$HOME/data/didactic/results/clustering_hparams_search/$method_path/$target/$model_id/$trial
end
end
end
end
end
end

# Running clustering evaluation of transformer encoder representations
for model in multimodal-xformer
for task in scratch scratch-cotrain finetune finetune-cotrain
for data in top-13 top-13+img all-clin all
for target in nt_probnp_group ht_severity
set method_path $model/$task/$data/$target
set method_name $model-$task-$data-$target
echo "Evaluating GMM clustering of $method_name models" >>$HOME/data/didactic/results/clustering_eval_$model.log 2>&1
python didactic/scripts/describe_representation_clustering.py (ls $HOME/data/didactic/results/clustering/$method_path/**/predictions.csv) --data_roots=$HOME/dataset/cardinal/v4/data --views A4C A2C --output_dir=$HOME/data/didactic/results/clustering_eval/$method_path >>$HOME/data/didactic/results/clustering_eval_$model.log 2>&1
end
end
end
end

# Running KNN evaluation of transformer encoder representations
for model in multimodal-xformer
for task in scratch scratch-cotrain finetune finetune-cotrain
for data in top-13 top-13+img all-clin all
for target in nt_probnp_group ht_severity
for ref_attr in ht_severity ht_grade
set method_path $model/$task/$data
set method_name $model-$task-$data-$target
echo "Evaluating KNN representations of $method_name models" >>$HOME/data/didactic/results/knn_eval_$model.log 2>&1
python didactic/scripts/describe_representation_knn.py (ls $HOME/data/didactic/models/$method_path/*$target*.ckpt) --data_roots $HOME/dataset/cardinal/v4/data --views A4C A2C --output_dir=$HOME/data/didactic/results/knn_eval/$method_path/$target '--neigh_kwargs={n_neighbors:8}' '--clinical_plot_kwargs={color:model}' --image_n_bins=8 --reference_attr=$ref_attr >>$HOME/data/didactic/results/knn_eval_$model.log 2>&1
end
end
end
end
end
