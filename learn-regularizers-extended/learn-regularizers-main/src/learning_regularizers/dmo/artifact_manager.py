"""
Artifact Path Manager

Centralizes all checkpoint and result paths for the DMO project.
Ensures reproducibility and organized artifact storage.
"""
from pathlib import Path
from datetime import datetime


def get_artifact_root():
    """Get the root artifact directory."""
    root = Path(__file__).resolve().parents[3] / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_checkpoints_dir(experiment="dmo"):
    """Get the checkpoints directory for an experiment."""
    ckpt_dir = get_artifact_root() / experiment / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_results_dir(experiment="dmo", result_type="json"):
    """Get the results directory for an experiment.
    
    Args:
        experiment: experiment name (default 'dmo')
        result_type: type of results (json, plots, metrics, etc.)
    """
    results_dir = get_artifact_root() / experiment / "results" / result_type
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_checkpoint_path(model_name, experiment="dmo"):
    """Get the full checkpoint file path.
    
    Args:
        model_name: name of the checkpoint (without .pt extension)
        experiment: experiment name (default 'dmo')
        
    Returns:
        Path object for the checkpoint file
    """
    ckpt_dir = get_checkpoints_dir(experiment)
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    return ckpt_dir / model_name


def get_result_file_path(filename, experiment="dmo", result_type="json"):
    """Get the full result file path.
    
    Args:
        filename: name of the result file
        experiment: experiment name (default 'dmo')
        result_type: type of results (json, plots, etc.)
        
    Returns:
        Path object for the result file
    """
    results_dir = get_results_dir(experiment, result_type)
    return results_dir / filename


def get_timestamped_result_file(base_name, experiment="dmo", result_type="json", extension="json"):
    """Get a timestamped result file path for reproducibility.
    
    Args:
        base_name: base name of the result file
        experiment: experiment name (default 'dmo')
        result_type: type of results (default 'json')
        extension: file extension (default 'json')
        
    Returns:
        Path object with timestamp appended before extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_name.endswith(f".{extension}"):
        filename = base_name.replace(f".{extension}", f"_{timestamp}.{extension}")
    else:
        filename = f"{base_name}_{timestamp}.{extension}"
    
    return get_result_file_path(filename, experiment, result_type)


def list_checkpoints(experiment="dmo"):
    """List all available checkpoints."""
    ckpt_dir = get_checkpoints_dir(experiment)
    return sorted([f.name for f in ckpt_dir.glob("*.pt")])


def list_results(experiment="dmo", result_type="json"):
    """List all available results of a given type."""
    results_dir = get_results_dir(experiment, result_type)
    return sorted([f.name for f in results_dir.glob("*")])


def print_artifact_structure(experiment="dmo"):
    """Print the artifact structure for an experiment."""
    artifact_root = get_artifact_root()
    exp_dir = artifact_root / experiment
    
    print(f"\n{'='*60}")
    print(f"Artifact Structure for {experiment}")
    print(f"{'='*60}")
    print(f"Root: {artifact_root}")
    print(f"\nExperiment: {exp_dir}")
    print(f"   checkpoints/ - Meta-optimizer checkpoints")
    print(f"     {list_checkpoints(experiment)[:3] if list_checkpoints(experiment) else '(empty)'}")
    print(f"   results/")
    print(f"      json/ - JSON results and metrics")
    print(f"      plots/ - Plot files")
    print(f"{'='*60}\n")
