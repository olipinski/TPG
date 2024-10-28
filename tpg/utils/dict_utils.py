"""Utils for dictionary manipulation."""


def default_to_regular(d):
    """
    Convert default dict to regular dict.

    Parameters
    ----------
    d: dict
        (Default)Dictionary to convert.

    Returns
    -------
    d: dict
        Dictionary converted to regular dict.

    """
    if isinstance(d, dict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d
