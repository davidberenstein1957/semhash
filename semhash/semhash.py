from __future__ import annotations

import numpy as np
from model2vec import StaticModel
from reach import Reach
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class SemHash:
    def __init__(self, model: SentenceTransformer | StaticModel) -> None:
        """Initialize SemHash."""
        self.model = model

    def deduplicate_embeddings(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray | None = None,
        threshold: float = 0.9,
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Deduplicate embeddings within one list or across two lists.

        :param embeddings1: Embeddings of the first list of texts.
        :param embeddings2: Optional embeddings of the second list of texts.
        :param threshold: Similarity threshold for deduplication.
        :param batch_size: Batch size for similarity computation.
        :return: Deduplicated indices and a mapping of duplicates to originals.
        """
        reach = Reach(vectors=embeddings1, items=[str(i) for i in range(len(embeddings1))])

        if embeddings2 is None:
            deduplicated_indices = set(range(len(embeddings1)))
            duplicate_to_original_mapping = {}

            results = reach.nearest_neighbor_indices(
                embeddings1, threshold=threshold, batch_size=batch_size, show_progressbar=True
            )

            for i, similar_indices in enumerate(tqdm(results, total=len(embeddings1))):
                if i not in deduplicated_indices:
                    continue  # Skip already marked duplicates

                # Remove the current index from similar items
                similar_indices = [sim_idx for sim_idx in similar_indices if sim_idx != i]

                for sim_idx in similar_indices:
                    if sim_idx in deduplicated_indices:
                        deduplicated_indices.remove(sim_idx)
                        duplicate_to_original_mapping[sim_idx] = i  # Map duplicate to original

            return np.array(list(deduplicated_indices)), duplicate_to_original_mapping
        else:
            deduplicated_indices_in_b = set()
            duplicate_to_original_mapping = {}

            results = reach.nearest_neighbor_indices(
                embeddings2, threshold=threshold, batch_size=batch_size, show_progressbar=True
            )

            for i, similar_indices in enumerate(tqdm(results, total=len(embeddings2))):
                if similar_indices.size == 0:
                    deduplicated_indices_in_b.add(i)
                else:
                    # Map to the first similar item in embeddings1
                    duplicate_to_original_mapping[i] = similar_indices[0]

            return np.array(list(deduplicated_indices_in_b)), duplicate_to_original_mapping

    def deduplicate(
        self,
        texts1: list[str],
        texts2: list[str] | None = None,
        threshold: float = 0.9,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Perform deduplication on one or two lists of texts.

        :param texts1: List of strings for the first dataset.
        :param texts2: Optional list of strings for the second dataset.
        :param threshold: Similarity threshold for deduplication.
        :return: Deduplicated indices and a mapping of duplicates to originals.
        """
        embeddings1 = self.model.encode(texts1, show_progressbar=True)
        embeddings2 = self.model.encode(texts2, show_progressbar=True) if texts2 else None

        deduplicated_indices, duplicate_mapping = self.deduplicate_embeddings(
            embeddings1, embeddings2=embeddings2, threshold=threshold
        )
        return deduplicated_indices, duplicate_mapping
