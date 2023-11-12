from typing import Hashable, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from vital.data.cardinal.config import CardinalTag, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import build_attributes_dataframe
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients
from vital.data.transforms import Interp1d


def build_clusterings_dataframe(
    patients: Patients, clusterings: Mapping[str, Mapping[Patient.Id, str]], cat_to_num: bool = False
) -> pd.DataFrame:
    """Builds a dataframe to store the data of the patients in each cluster, repeating patients across clusterings.

    Args:
        patients: Collection of patients to include in the dataframe.
        clusterings: Instances of clustering of the patients population, represented as mappings between patient IDs
            and cluster labels.
        cat_to_num: Whether to convert the categorical attributes to numerical labels, based on the order of the
            categories.

    Returns:
        Dataframe, with a multi-index with levels `(model, cluster, patient_id)`, containing the data of the patients
        from each cluster for each model/clustering, repeating patients as necessary.
    """
    # Convert clusterings from mapping between item IDs and cluster IDs to lists of patient IDs by cluster
    clusterings = {
        clustering_id: {
            cluster_label: sorted(
                patient_id for patient_id, patient_cluster in clusters.items() if patient_cluster == cluster_label
            )
            for cluster_label in sorted(set(clusters.values()))
        }
        for clustering_id, clusters in clusterings.items()
    }

    data = patients.to_dataframe()

    if cat_to_num:
        # Convert the categorical attributes to numerical labels
        def _to_num(attr_data: pd.Series) -> pd.Series:
            if attr_data.dtype == "category":
                attr_data = attr_data.cat.codes
            return attr_data

        data = data.apply(_to_num)

    # For each clustering, extract the data of the patients in each cluster
    clusterings_data = pd.concat(
        {
            clustering_id: pd.concat(
                {
                    cluster_label: data.loc[patient_ids_in_cluster]
                    for cluster_label, patient_ids_in_cluster in clusters.items()
                }
            )
            for clustering_id, clusters in clusterings.items()
        },
        names=["model", "cluster", "patient_id"],
    )

    return clusterings_data


def build_knn_dataframe(patients: Patients, kneighbors: np.ndarray, cat_to_num: bool = False) -> pd.DataFrame:
    """Builds a dataframe to store the data of the nearest neighbors of each patient, repeating patients as necessary.

    Args:
        patients: Collection of patients to include in the dataframe.
        kneighbors: Array (of `Patient.Id`s) of shape `(n_encodings, n_patients, n_neighbors)` containing the IDs of the
            nearest neighbors of each patient for each encoding.
        cat_to_num: Whether to convert the categorical attributes to numerical labels, based on the order of the
            categories.

    Returns:
        Dataframe, with a multi-index with levels `(model, patient_id, neighbor_id)`, containing the data of the nearest
        neighbors of each patient for each model, repeating patients as necessary.
    """
    data = patients.to_dataframe()

    if cat_to_num:
        # Convert the categorical attributes to numerical labels
        def _to_num(attr_data: pd.Series) -> pd.Series:
            if attr_data.dtype == "category":
                attr_data = attr_data.cat.codes
            return attr_data

        data = data.apply(_to_num)

    # For each encoding, extract the data of the nearest neighbors of each patient
    neigh_data = pd.concat(
        {
            f"{enc_idx}": pd.concat(
                {
                    patient_id: data.loc[kneighbors_ids]
                    for patient_id, kneighbors_ids in zip(patients, enc_kneighbors_ids)
                }
            )
            for enc_idx, enc_kneighbors_ids in enumerate(kneighbors)
        },
        names=["model", "patient_id", "neighbor_id"],
    )

    return neigh_data


def build_time_series_attr_by_patient_group_dataframe(
    patients_groups: Mapping[Hashable, Iterable[Patient]],
    attr: Tuple[ViewEnum, TimeSeriesAttribute],
    group_desc: str = "group",
    mask_tag: str = CardinalTag.mask,
    resampling_rate: int = 128,
) -> pd.DataFrame:
    """Builds a dataframe with the average curve of a time-series attribute by patient group.

    Args:
        patients_groups: Mapping between group ID and the patients in that group.
        attr: A pair of view and time-series attribute to compute the average curve of.
        group_desc: Description of the semantic meaning of the groups.
        mask_tag: Tag of the segmentation mask for which to extract the time-series attribute data.
        resampling_rate: Number of points at which to resample the time-series attribute curves from each patient, so
            that they can be easily compared and aggregated together.

    Returns:
        Dataframe with the average curve of an time-series attribute by patient group, in long format.
    """
    resampling_fn = Interp1d(resampling_rate)

    # For each group, compute the attribute's average curve
    data = {}
    for group, patients in patients_groups.items():
        # When stacking the attributes from all patients in a group, it's necessary to resample the attributes
        # Otherwise the variable number of frames by seq would cause errors because of unequal array shapes
        group_attr_data = np.vstack(
            [resampling_fn(patient.get_mask_attributes(mask_tag)[attr[0]][attr[1]]) for patient in patients]
        )
        # Add the unnecessary nested dict level to conform to the API expected by `build_attributes_dataframe`
        data[group] = {attr: group_attr_data.mean(axis=0)}

    # Structure the attribute's average curve by group as a dataframe in long format
    # to make it easier to use seaborn's plot
    return build_attributes_dataframe(data, data_name=group_desc)
