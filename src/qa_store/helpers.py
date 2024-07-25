import json

from loguru import logger


def get_json_list(json_data: str) -> list:
    """
    Get a list from a JSON object.

    Parameters
    ----------
    json_data : dict
        The JSON object containing a single key-value
        pair, where the value is a list.

    Returns
    -------
    list
        The list extracted from the JSON object.

    Raises
    ------
    ValueError
        If the JSON data is not a dictionary or the
        dictionary does not contain a list.

    Notes
    -----
    This function assumes that the input JSON object
    has only one key-value pair, and the value
    associated with that key is a list.

    Examples
    --------
    >>> json_data = {"items": [1, 2, 3, 4, 5]}
    >>> result = get_json_list(json_data)
    >>> print(result)
    [1, 2, 3, 4, 5]

    >>> invalid_json = {"items": "not a list"}
    >>> get_json_list(invalid_json)
    Raises ValueError: The JSON data does not contain a list.
    """
    json_data = json.loads(json_data)
    if not isinstance(json_data, dict):
        raise ValueError("The JSON data is not a dictionary.")
    key = next(iter(json_data))
    if not isinstance(json_data[key], list):
        logger.warning("The JSON data does not contain a list.")
        return [json_data]
    return json_data[key]
