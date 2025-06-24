#!/bin/bash

CARDINAL_DATA_PATH=$1
MODEL_COMMON_PATH=$2
OUTPUT_DIR=$3

# Compute the alignment scores between the different splits of the best configuration
# IMPORTANT: Requires the model to be registered in the Comet model registry w/ the "-{split_idx}" suffix
echo "Computing alignment scores between models" >>$OUTPUT_DIR/score_models_alignment.log 2>&1
python ~/remote/didactic/didactic/scripts/score_models_alignment.py "$MODEL_COMMON_PATH-0" "$MODEL_COMMON_PATH-1" "$MODEL_COMMON_PATH-2" "$MODEL_COMMON_PATH-3" "$MODEL_COMMON_PATH-4" --data_roots $CARDINAL_DATA_PATH --views A4C A2C --output_file=$OUTPUT_DIR/alignment_scores.csv >>$OUTPUT_DIR/score_models_alignment.log 2>&1

# For each of the model
for split_idx in $(seq 0 4); do

  # Copy the dataset partitioning files to "Subset" sub-directories (for plotting purposes)
  mkdir -p $OUTPUT_DIR/split_idx=$split_idx/Subset
  rsync -av $CARDINAL_DATA_PATH/splits/$split_idx/ $OUTPUT_DIR/split_idx=$split_idx/Subset

  # Split patients into bins w.r.t. continuum param predicted for each patient
  echo "Splitting patients into bins w.r.t. continuum param for model on split $split_idx" >>$OUTPUT_DIR/group_patients_by_continuum_param.log 2>&1
  python ~/remote/didactic/didactic/scripts/group_patients_by_predictions.py "$MODEL_COMMON_PATH-$split_idx" --data_roots $CARDINAL_DATA_PATH --views A4C A2C --bins=6 --output_dir=$OUTPUT_DIR/split_idx=$split_idx/continuum_param_bins range --bounds 0 1 >>$OUTPUT_DIR/group_patients_by_continuum_param.log 2>&1

  # Describe the tabular attributes of the patients in each bin
  echo "Describing tabular attributes of patients in each bin for model on split $split_idx" >>$OUTPUT_DIR/describe_patients_by_bins.log 2>&1
  python ~/remote/didactic/didactic/scripts/describe_patients.py --data_roots $CARDINAL_DATA_PATH --views A4C A2C --subsets $(find $OUTPUT_DIR/split_idx=$split_idx/continuum_param_bins -name "*.txt" | sort | tr "\n" " ") --output_dir=$OUTPUT_DIR/split_idx=$split_idx/describe_patients_by_bins >>$OUTPUT_DIR/describe_patients_by_bins.log 2>&1

  # Plot variability of time-series attributes w.r.t. predicted continuum
  echo "Plotting variability of attrs w.r.t. continuum predicted by model on split $split_idx" >>$OUTPUT_DIR/attrs_wrt_continuum_bins.log 2>&1
  python ~/remote/didactic/vital/vital/data/cardinal/plot_attrs_wrt_groups.py --data_roots $CARDINAL_DATA_PATH --views A4C A2C --groups_txt $(find $OUTPUT_DIR/split_idx=$split_idx/continuum_param_bins -name "*.txt" | sort | tr "\n" " ") '--time_series_plot_kwargs={errorbar:,palette:flare}' --output_dir=$OUTPUT_DIR/split_idx=$split_idx/attrs_wrt_continuum_param >>$OUTPUT_DIR/attrs_wrt_continuum_param.log 2>&1

  # Plot 2D embeddings of the transformer encoder representations
  echo "Generating 2D embedding of latent space for model on split $split_idx, using PaCMAP" >>$OUTPUT_DIR/2d_embeddings.log 2>&1
  python ~/remote/didactic/didactic/scripts/cardiac_multimodal_representation_plot.py "$MODEL_COMMON_PATH-$split_idx" --data_roots $CARDINAL_DATA_PATH --views A4C A2C --plot_categorical_attrs_dirs $OUTPUT_DIR/split_idx=$split_idx/continuum_param_bins $OUTPUT_DIR/split_idx=$split_idx/Subset '--cat_plot_kwargs={style:Subset}' '--num_plot_kwargs={style:Subset,palette:flare}' --output_dir=$OUTPUT_DIR/split_idx=$split_idx/2d_embeddings >>$OUTPUT_DIR/2d_embeddings.log 2>&1

  # Plot variability of predicted continuum w.r.t. position along the continuum
  echo "Plotting variability of predicted continuum w.r.t. position along the continuum between models" >>$OUTPUT_DIR/plot_models_variability.log 2>&1
  python ~/remote/didactic/didactic/scripts/plot_models_variability.py "$MODEL_COMMON_PATH-0" "$MODEL_COMMON_PATH-1" "$MODEL_COMMON_PATH-2" "$MODEL_COMMON_PATH-3" "$MODEL_COMMON_PATH-4" --data_roots $CARDINAL_DATA_PATH --views A4C A2C --plot_categorical_attrs_dirs $OUTPUT_DIR/split_idx=$split_idx/Subset '--plot_kwargs={hue:ht_severity,style:Subset}' --output_dir=$OUTPUT_DIR/split_idx=$split_idx/continuum_param_variability >>$OUTPUT_DIR/plot_models_variability.log 2>&1

done
