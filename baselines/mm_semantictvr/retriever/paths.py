from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "baseline"
DEFAULT_CANDIDATE_ROOT = REPO_ROOT / "candidates" / "baseline"


INDEX_PREFIX_MAP = {
    "msrvtt": "msrvtt",
    "actnet": "activitynet",
    "didemo": "didemo",
    "lsmdc": "lsmdc",
}


CAPTION_PREFIX_MAP = {
    "msrvtt": "msrvtt",
    "actnet": "actnet",
    "didemo": "didemo",
    "lsmdc": "lsmdc",
}


@dataclass(frozen=True)
class RetrieverPaths:
    dataset_dir: Path
    caption_dir: Path
    output_root: Path
    candidate_output_root: Path
    train_caption_file: Optional[Path]
    train_addition_caption_file: Optional[Path]
    test_caption_file: Path
    train_data_file: Optional[Path]
    train_index_file: Path
    test_index_file: Path
    codebook_embedding_file: Path


def _resolve_root(path_str: Optional[str], default_root: Path) -> Path:
    if not path_str:
        return default_root
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def get_index_prefix(dataset_name: str) -> str:
    return INDEX_PREFIX_MAP.get(dataset_name, dataset_name)


def get_caption_prefix(dataset_name: str) -> str:
    return CAPTION_PREFIX_MAP.get(dataset_name, dataset_name)


def get_index_subdir(
    index_type: str,
    code_book_size: int,
    code_book_num: int,
    version: str,
) -> str:
    if index_type == "videorqvae":
        return f"videorqvae_v{version}_c{code_book_size}_l{code_book_num}"
    return f"{index_type}_c{code_book_size}_l{code_book_num}"


def build_retriever_paths(
    dataset_name: str,
    mode: str,
    index_type: str,
    code_book_size: int,
    code_book_num: int,
    version: str,
    output_root: Optional[str] = None,
    candidate_output_root: Optional[str] = None,
) -> RetrieverPaths:
    dataset_dir = DATA_ROOT / dataset_name / mode
    caption_dir = DATA_ROOT / dataset_name / "video_retreival_caption"
    caption_prefix = get_caption_prefix(dataset_name)
    index_prefix = get_index_prefix(dataset_name)
    index_subdir = get_index_subdir(index_type, code_book_size, code_book_num, version)
    index_root = dataset_dir / index_subdir

    train_caption_file = caption_dir / f"{caption_prefix}_ret_train.json"
    train_addition_caption_file = caption_dir / f"{caption_prefix}_ret_train_addition.json"
    test_caption_file = caption_dir / f"{caption_prefix}_ret_test.json"
    if dataset_name == "lsmdc" and not test_caption_file.exists():
        test_caption_file = caption_dir / f"{caption_prefix}_ret_test_1000.json"

    train_data_file: Optional[Path] = None
    if index_type == "videorqvae":
        train_index_file = index_root / f"{index_prefix}_videorqvae_index_train.json"
        train_data_file = train_index_file
        test_index_file = index_root / f"{index_prefix}_videorqvae_index_test.json"
    else:
        train_index_file = index_root / f"{index_prefix}_index_internvideo2_emb_train.json"
        test_index_file = index_root / f"{index_prefix}_index_internvideo2_emb_test.json"

    return RetrieverPaths(
        dataset_dir=dataset_dir,
        caption_dir=caption_dir,
        output_root=_resolve_root(output_root, DEFAULT_OUTPUT_ROOT),
        candidate_output_root=_resolve_root(candidate_output_root, DEFAULT_CANDIDATE_ROOT),
        train_caption_file=train_caption_file,
        train_addition_caption_file=train_addition_caption_file,
        test_caption_file=test_caption_file,
        train_data_file=train_data_file,
        train_index_file=train_index_file,
        test_index_file=test_index_file,
        codebook_embedding_file=index_root / "codebook_embedding.pt",
    )
