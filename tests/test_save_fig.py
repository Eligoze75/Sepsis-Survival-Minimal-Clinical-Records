# test codes are generated using Chatgpt

import pytest
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_fig import save_fig

@pytest.fixture
def sample_plot():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    yield fig
    plt.close(fig) 

@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path

# -------------------------------
# Expected use cases
# -------------------------------
def test_save_fig_success(sample_plot, temp_directory):
    filename = "test_plot"
    save_fig(temp_directory, filename, extension="png")
    
    file_path = os.path.join(temp_directory, f"{filename}.png")
    assert os.path.isfile(file_path)

def test_save_fig_other_extensions(sample_plot, temp_directory):
    for ext in ["jpg", "jpeg", "pdf", "svg"]:
        filename = f"test_plot_{ext}"
        save_fig(temp_directory, filename, extension=ext)
        file_path = os.path.join(temp_directory, f"{filename}.{ext}")
        assert os.path.isfile(file_path)

# -------------------------------
# Error cases
# -------------------------------
def test_save_fig_invalid_extension(sample_plot, temp_directory):
    with pytest.raises(ValueError, match="Extension must be one of"):
        save_fig(temp_directory, "test_plot_invalid", extension="txt")

def test_save_fig_nonexistent_directory(sample_plot):
    non_existent_directory = "/nonexistent_directory"
    with pytest.raises(FileNotFoundError, match=f"Directory {non_existent_directory} does not exist."):
        save_fig(non_existent_directory, "test_plot")

def test_save_fig_invalid_filename_type(sample_plot, temp_directory):
    with pytest.raises(TypeError, match="Filename must be a string"):
        save_fig(temp_directory, 123)  # filename is not a string

def test_save_fig_invalid_directory_type(sample_plot):
    with pytest.raises(TypeError, match="Directory must be a string"):
        save_fig(123, "test_plot")  # directory is not a string

# -------------------------------
# Edge cases
# -------------------------------
def test_save_fig_empty_filename(sample_plot, temp_directory):
    empty_filename = ""
    with pytest.raises(ValueError, match="Filename must be a non-empty string"):
        save_fig(temp_directory, empty_filename)

def test_save_fig_uppercase_extension(sample_plot, temp_directory):
    filename = "test_plot_uppercase"
    save_fig(temp_directory, filename, extension="PNG")  # uppercase extension
    
    file_path = os.path.join(temp_directory, f"{filename}.png")  # saved as lowercase
    assert os.path.isfile(file_path)
