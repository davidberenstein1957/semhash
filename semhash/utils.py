from difflib import ndiff


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
