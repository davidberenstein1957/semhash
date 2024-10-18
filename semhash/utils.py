from difflib import ndiff
from typing import Iterator

import numpy as np
import numpy.typing as npt
from reach import Reach
from tqdm import tqdm


class CustomReach(Reach):
    def indices_threshold(
        self,
        vectors: npt.NDArray,
        threshold: float = 0.5,
        batch_size: int = 100,
        show_progressbar: bool = False,
    ) -> Iterator[np.ndarray]:
        """
        Retrieve the unsorted indices of the nearest neighbors for the given vectors above a threshold.

        This method is useful if you want to find the indices of the most similar items above a certain threshold
        but don't need them to be sorted, which is a time consuming operation.

        :param vectors: The vectors to find the most similar vectors to.
        :param threshold: The threshold to use.
        :param batch_size: The batch size to use. 100 is a good default option. Increasing the batch size may increase
            the speed.
        :param show_progressbar: Whether to show a progressbar.

        :return: For each item in the input, the indices of the nearest neighbors are returned as an array.
        """
        vectors = np.array(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        return self._indices_threshold_batch(vectors, batch_size, threshold, show_progressbar)

    def _indices_threshold_batch(
        self,
        vectors: np.ndarray,
        batch_size: int,
        threshold: float,
        show_progressbar: bool,
    ) -> Iterator[np.ndarray]:
        """Batched cosine similarity. Returns the indices of vectors above the threshold."""
        for i in tqdm(range(0, len(vectors), batch_size), disable=not show_progressbar):
            batch = vectors[i : i + batch_size]
            similarities = self._sim(batch, self.norm_vectors)
            for _, sims in enumerate(similarities):
                # Yield indices of neighbors with similarity above the threshold
                yield np.flatnonzero(sims > threshold)


def display_word_differences(x: str, y: str) -> str:
    """
    Display the word-level differences between two texts.

    :param x: First text.
    :param y: Second text.
    :return: A string showing word-level differences, wrapped in a code block.
    """
    diff = ndiff(x.split(), y.split())
    formatted_diff = "\n".join(word for word in diff if word.startswith(("+", "-")))
    return f"```\n{formatted_diff}\n```"
