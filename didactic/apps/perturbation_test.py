import logging
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from vital.data.cardinal.config import CardinalTag, TabularAttribute
from vital.data.cardinal.datapipes import process_patient
from vital.data.cardinal.utils.itertools import Patients
from vital.utils.format.torch import numpy_to_torch

from didactic.models.explain import SelfAttentionGenerator
from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask

logger = logging.getLogger(__name__)


def compute_attributes_relevance(
    model: CardiacMultimodalRepresentationTask,
    patients: Patients,
    relevancy_target: TabularAttribute,
    ignore_attrs: Sequence[TabularAttribute] = None,
    mask_tag: str = CardinalTag.mask,
    progress_bar: bool = False,
) -> pd.DataFrame:
    """Computes the relevance of each input attribute, averaged over all patients.

    Args:
        model: Transformer encoder model to use for inference.
        patients: Patients on which to compute the relevance.
        relevancy_target: Target attribute w.r.t. which to compute the relevancy computation.
        ignore_attrs: Attributes to ignore, i.e. mask, when computing the relevance.
        mask_tag: Tag of the segmentation mask for which to extract the time-series attributes.
        progress_bar: If ``True``, enables progress bars detailing the progress of the processing and encoding patients
            data.

    Returns:
        A dataframe containing the relevance of each attribute, for each patient.
    """
    tab_attrs, time_series_attrs = model.hparams.tabular_attrs, model.hparams.time_series_attrs
    patients_data = (
        process_patient(
            patient,
            tabular_attrs=tab_attrs + (relevancy_target,),
            time_series_attrs=time_series_attrs,
            mask_tag=mask_tag,
            mask_attrs=ignore_attrs,
        )
        for patient in patients.values()
    )

    relevancy_gen = SelfAttentionGenerator(model)
    attr_tags = model.token_tags[:-1]  # Exclude the CLS token

    relevancy_data = {}
    for patient_data in tqdm(
        patients_data,
        desc="Collecting attention maps from patients",
        unit="patient",
        total=len(patients),
        disable=not progress_bar,
        leave=False,
    ):
        # Separate the tabular and time-series attributes + add a batch dimension
        patient_tab_attrs = numpy_to_torch({attr: patient_data[attr][None, ...] for attr in tab_attrs})
        patient_time_series_attrs = numpy_to_torch(
            {
                (view, attr): patient_data[view][attr][None, ...]
                for view in model.hparams.views
                for attr in time_series_attrs
            }
        )
        target_label = torch.from_numpy(patient_data[relevancy_target][None, ...])

        relevancy_by_attr = (
            relevancy_gen.generate_relevancy(
                patient_tab_attrs, patient_time_series_attrs, target_label, relevancy_target
            )[0]
            .cpu()
            .numpy()
        )

        relevancy_data[patient_data["id"]] = {
            attr: attr_relevancy for attr, attr_relevancy in zip(attr_tags, relevancy_by_attr)
        }

    return pd.DataFrame(relevancy_data)


def run_perturbation_test(
    model: CardiacMultimodalRepresentationTask,
    patients: Patients,
    target: TabularAttribute,
    perturbation_mode: Literal["negative", "positive"] = "negative",
    manual_perturbations: Sequence[TabularAttribute] = None,
    mask_tag: str = CardinalTag.mask,
    progress_bar: bool = False,
    save_dir: Path = None,
) -> pd.Series:
    """Performs perturbation tests and computes model's AUROC score the more attributes are removed.

    Args:
        model: Transformer encoder model to use for inference.
        patients: Patients on which to compute measure the model's AUROC score.
        target: Target attribute w.r.t. which to compute the model's AUROC score.
        perturbation_mode: Type of perturbation test to perform. If ``negative``, the test will remove the least
            relevant attributes first. If ``positive``, the test will remove the most relevant attributes first.
        manual_perturbations: Attributes to manually perturb before automatically selecting following attributes.
        mask_tag: Tag of the segmentation mask for which to extract the time-series attributes.
        progress_bar: If ``True``, enables progress bars detailing the progress of how many attributes are left to
            perturb.
        save_dir: If provided, directory where to save intermediate relevance upon removing each new attribute.

    Returns:
        A series containing the AUROC score of the model for each further attribute removed.
    """
    if manual_perturbations is None:
        manual_perturbations = []

    tab_attrs, time_series_attrs = model.hparams.tabular_attrs, model.hparams.time_series_attrs
    n_attrs = len(tab_attrs) + (len(time_series_attrs) * len(model.hparams.views))

    attrs_to_remove = []
    attrs_perturbation_scores = {}
    with tqdm(
        total=n_attrs + 1, desc="Removing gradually more attributes", unit="attr", disable=not progress_bar
    ) as pbar:
        while True:
            n_removed_attrs = len(attrs_to_remove)

            attrs_relevancy = compute_attributes_relevance(
                model, patients, target, ignore_attrs=attrs_to_remove, mask_tag=mask_tag, progress_bar=progress_bar
            )
            attrs_relevancy = attrs_relevancy.mean(axis="columns")  # Average the relevancy over all patients
            attrs_relevancy = attrs_relevancy.sort_values(ascending=perturbation_mode == "negative")
            attrs_relevancy = attrs_relevancy.to_frame(name="Relevancy")
            attrs_relevancy["is_masked"] = attrs_relevancy.index.isin(attrs_to_remove)

            if save_dir:
                step_name = f"step_{n_removed_attrs}"
                if attrs_to_remove:
                    step_name += f"={attrs_to_remove[-1].replace('/', '_')}"
                (save_dir / step_name).mkdir(parents=True, exist_ok=True)
                attrs_relevancy.to_csv(save_dir / step_name / "attributes_relevance.csv", index_label="Attribute")

            patients_data = (
                process_patient(
                    patient,
                    tabular_attrs=tab_attrs + (target,),
                    time_series_attrs=time_series_attrs,
                    mask_tag=mask_tag,
                    mask_attrs=attrs_to_remove,
                )
                for patient in patients.values()
            )

            pred_logits = {}
            target_labels = {}
            for patient_data in patients_data:
                # Separate the tabular and time-series attributes + add a batch dimension
                patient_tab_attrs = numpy_to_torch({attr: patient_data[attr][None, ...] for attr in tab_attrs})
                patient_time_series_attrs = numpy_to_torch(
                    {
                        (view, attr): patient_data[view][attr][None, ...]
                        for view in model.hparams.views
                        for attr in time_series_attrs
                    }
                )

                # Compute the predicted probabilities for each class for the target attribute
                pred_logits[patient_data["id"]] = (
                    model(patient_tab_attrs, patient_time_series_attrs, task="predict")[target][0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                target_labels[patient_data["id"]] = patient_data[target].item()

            # Compute the ROC AUC score for the target attribute
            pred_logits = pd.DataFrame.from_dict(pred_logits, orient="index")
            target_labels = pd.Series(target_labels)
            attrs_perturbation_scores[attrs_to_remove[-1] if attrs_to_remove else "none"] = roc_auc_score(
                target_labels, softmax(pred_logits.to_numpy(), axis=1), multi_class="ovr"
            )

            pbar.update(1)

            # Select the attribute occurring first (i.e. least/most relevant) that has not already been removed
            # This is done because the absence (i.e. masking) of attributes could be considered relevant by the model,
            # so we have to look at available attributes to make sure we remove new ones
            remaining_attrs = attrs_relevancy[~attrs_relevancy.index.isin(attrs_to_remove)]
            if remaining_attrs.empty:
                break

            if manual_perturbations:
                next_attr = manual_perturbations.pop(0)
            else:
                next_attr = remaining_attrs.index[0]

            attrs_to_remove.append(next_attr)

    return pd.Series(attrs_perturbation_scores, name="AUROC")


def main():
    """Run the script."""
    import argparse

    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt
    from vital.utils.logging import configure_logging
    from vital.utils.saving import load_from_checkpoint

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pretrained_encoder",
        type=Path,
        help="Path to a model checkpoint, or name of a model from a Comet model registry, of an encoder",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--relevancy_target",
        type=TabularAttribute,
        default=TabularAttribute.ht_severity,
        help="Target attribute w.r.t. which to compute the relevancy computation",
    )
    parser.add_argument(
        "--perturbation_mode",
        type=str,
        choices=["negative", "positive"],
        default="negative",
        help="Type of perturbation test to perform. 'negative' removes the least relevant attributes first, "
        "'positive' removes the most relevant attributes first",
    )
    parser.add_argument(
        "--manual_perturbations",
        type=TabularAttribute,
        nargs="*",
        help="Attributes to manually perturb before automatically selecting following attributes",
    )
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument("--output_dir", type=Path, default=Path.cwd(), help="Directory to save the output files")
    args = parser.parse_args()
    kwargs = vars(args)

    encoder_ckpt, relevancy_target, perturbation_mode, manual_perturbations, mask_tag, output_dir = (
        kwargs.pop("pretrained_encoder"),
        kwargs.pop("relevancy_target"),
        kwargs.pop("perturbation_mode"),
        kwargs.pop("manual_perturbations"),
        kwargs.pop("mask_tag"),
        kwargs.pop("output_dir"),
    )

    encoder = load_from_checkpoint(encoder_ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask)
    patients = Patients(**kwargs)

    # Run the negative perturbation test
    attrs_perturbation_scores = run_perturbation_test(
        encoder,
        patients,
        relevancy_target,
        perturbation_mode=perturbation_mode,
        manual_perturbations=manual_perturbations,
        mask_tag=mask_tag,
        progress_bar=True,
        save_dir=output_dir,
    )

    # Save the results to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    attrs_perturbation_scores.to_csv(output_dir / "perturbation_test.csv", index_label="Attribute")

    # Plot the AUROC curve w.r.t. the number of attributes removed
    # Ensure that matplotlib is using 'agg' backend in non-interactive case
    plt.switch_backend("agg")

    attrs_perturbation_scores = attrs_perturbation_scores.to_frame()
    with sns.axes_style("darkgrid"):
        attrs_perturbation_scores["% of attributes removed"] = np.linspace(0, 100, len(attrs_perturbation_scores))
        sns.lineplot(data=attrs_perturbation_scores, x="% of attributes removed", y="AUROC")
    plt.savefig(output_dir / "perturbation_test.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
