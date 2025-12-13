import matplotlib.pyplot as plt
import os

def save_fig(directory: str, filename: str, extension: str = "png", dpi: int = 300, 
             bbox_inches: str ="tight", transparent: bool = True):
    """
    Save a Matplotlib figure to a file of a specified type (PNG, JPEG, PDF, or SVG) in the specified
    directory.

    Parameters
    ----------
    directory : str
        The directory where the file will be saved.
    filename : str
        The name of the file without extension
    extension : str, optional
        The file extension/type (e.g., 'png', 'jpg', 'pdf', 'svg'). Default is 'png'.
    dpi : int, optional
        The resolution of saved figure. Default is 300.
    bbox_inches : str, optional
        Controls how the bounding box is handled. Default is tight.
    transparent : bool, optional
        Whether the saved figure uses a transparent background. Default is True.

    Raises
    ------
    TypeError
        If the directory or filename is not a string.
    ValueError
        If the filename is not empty.
    ValueError
        If the extension is not one of: "png", "jpg", "jpeg", "pdf", or "svg".
    FileNotFoundError
        If the specified directory does not exist.
    """
    extension = extension.lower()
    valid_extensions = ("png", "jpg", "jpeg", "pdf", "svg")
    
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")
    if filename == "":
        raise ValueError("Filename must be a non-empty string")
    if not isinstance(directory, str):
        raise TypeError("Directory must be a string")
    if extension not in valid_extensions:
        raise ValueError(f"Extension must be one of {valid_extensions}")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")


    filepath = os.path.join(directory, f"{filename}.{extension}")
    
    plt.savefig(
        filepath,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent
    )

