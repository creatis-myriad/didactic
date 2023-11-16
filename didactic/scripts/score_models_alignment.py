from typing import Dict

import numpy as np
import pandas as pd


def score_embeddings_alignment(embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Computes the alignment score of each embedding w.r.t. the average of the set.

    The alignment score is defined as the average over the Euclidean distance between points in each embedding and their
    average embedding (across the different embeddings).

    Args:
        embeddings: Mapping between the name of each embedding and the corresponding (N, [E]) embedding matrix.

    Returns:
        The alignment score of each embedding.
    """
    # Stack the embeddings into a single array,
    # making sure to expand the embedding dimension to 1 if it is missing, i.e. for 1D embeddings
    embeddings_arr = np.stack([emb.reshape((len(emb), -1)) for emb in embeddings.values()], axis=1)  # (N, M, E)

    # Compute the reference embedding as the average of the embeddings
    mean_embedding = np.mean(embeddings_arr, axis=1, keepdims=True)  # (N, 1, E)

    # Compute the Euclidean distance between the items in each embedding and the reference embedding
    distances = np.linalg.norm(embeddings_arr - mean_embedding, axis=-1)  # (N, M)

    # For each embedding, compute the alignment score as the average distance to the reference embedding
    aligment_scores = distances.mean(axis=0)  # (M,)

    return {emb_name: score for emb_name, score in zip(embeddings.keys(), aligment_scores)}


def main():
    """Run the script."""

    import pprint
    from argparse import ArgumentParser
    from pathlib import Path

    from tqdm.auto import tqdm
    from vital.data.cardinal.config import CardinalTag
    from vital.data.cardinal.utils.itertools import Patients
    from vital.utils.saving import load_from_checkpoint

    from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
    from didactic.tasks.utils import encode_patients

    parser = ArgumentParser("Script to score how well models align with the average of an ensemble of similar models")
    parser.add_argument(
        "pretrained_encoder",
        nargs="+",
        type=Path,
        help="Path to model checkpoint, or name of a model from a Comet model registry, of an encoder",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument(
        "--encoding_task",
        type=str,
        default="unimodal_param",
        choices=["encode", "unimodal_param"],
        help="Encoding task used to generate the embeddings for the computation of the alignment score",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path to the CSV file where to save the alignment scores",
    )
    args = parser.parse_args()

    if len(args.pretrained_encoder) < 3:
        raise ValueError("At least 3 models are required to compute a meaningful alignment score")

    kwargs = vars(args)
    pretrained_encoders, mask_tag, encoding_task, output_file = list(
        map(kwargs.pop, ["pretrained_encoder", "mask_tag", "encoding_task", "output_file"])
    )

    # Compute the embeddings of the patients for each model
    # Load the models directly inside the dict comprehension so that (hopefully) only one model gets loaded into GPU
    # memory at a time, to avoid having all models loaded at any given time
    patients = Patients(**kwargs)
    embeddings = {
        # Take the first target as the prediction (indexing of the dict of predictions returned by `encode_patients`)
        ckpt.stem: encode_patients(
            encoder := load_from_checkpoint(ckpt, expected_checkpoint_type=CardiacMultimodalRepresentationTask),
            patients.values(),
            mask_tag=mask_tag,
            progress_bar=True,
            task=encoding_task,
        )[encoder.hparams.target_tabular_attrs[0]]
        for ckpt in tqdm(pretrained_encoders, desc="Computing embeddings for the patients", unit="model")
    }

    # Compute the alignment scores
    alignment_scores = score_embeddings_alignment(embeddings)

    # Save the alignment scores to CSV or display them to the terminal
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_dict(alignment_scores, orient="index", columns=["alignment_score"]).to_csv(output_file)
    else:
        print("Alignment scores of each model:")
        pprint.pprint(alignment_scores)


if __name__ == "__main__":
    main()
