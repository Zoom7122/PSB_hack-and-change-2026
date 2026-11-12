"""Build the flat ZIP required by the PSB automatic evaluation platform."""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "psb_hack_change_submission.zip"
MODULE_PATHS = (
    "API/backend_api.py",
    "API/for_solutions.py",
    "answer_generation/models.py",
    "answer_generation/service.py",
    "answer_generation/verification.py",
    "data_processing/chunking.py",
    "data_processing/config.py",
    "data_processing/contract_adapter.py",
    "data_processing/hybrid.py",
    "data_processing/ingest.py",
    "data_processing/llm_client.py",
    "data_processing/refs.py",
    "data_processing/search.py",
    "data_processing/textproc.py",
    "data_processing/vector_index.py",
    "query_processing_module/query_processing/config.py",
    "query_processing_module/query_processing/contracts.py",
    "query_processing_module/query_processing/llm_client.py",
    "query_processing_module/query_processing/multi_stage.py",
    "query_processing_module/query_processing/prompts.py",
    "query_processing_module/query_processing/service.py",
    "query_processing_module/query_processing/validator.py",
    "rag_pipeline/corpus.py",
    "rag_pipeline/indexing.py",
    "rag_pipeline/output.py",
    "rag_pipeline/retrieval.py",
)
REQUIREMENTS = (
    "httpx==0.28.1",
    "numpy==2.2.6",
    "openai==3.45.0",
    "pydantic==2.14.2",
    "python-dotenv==1.1.1",
    "rank-bm25==0.2.2",
)
ENTRYPOINT = '''"""Single-file-layout entry point for PSB platform checks."""

from _psb_api_for_solutions import main


if __name__ == "__main__":
    main()
'''
FIRST_PARTY_PACKAGES = (
    "API",
    "answer_generation",
    "data_processing",
    "query_processing_module",
    "rag_pipeline",
)


def _module_map() -> dict[str, str]:
    """Map source package names to unique flat Python module names."""
    return {
        Path(path).with_suffix("").as_posix().replace("/", "."):
        "_psb_" + Path(path).with_suffix("").as_posix().replace("/", "_").lower()
        for path in MODULE_PATHS
    }


def _relative_name(current: str, level: int, module: str | None) -> str:
    """Resolve a relative import to its original absolute module name."""
    package = current.split(".")[:-1]
    prefix = package[:len(package) - level + 1]
    suffix = module.split(".") if module else []
    return ".".join(prefix + suffix)


def _rewrite_import(node: ast.AST, current: str, modules: dict[str, str]) -> str | None:
    """Rewrite one internal import so it targets a flat module file."""
    if isinstance(node, ast.ImportFrom):
        target = _relative_name(current, node.level, node.module) if node.level else node.module
        package = current.rpartition(".")[0]
        if target and target not in modules and f"{package}.{target}" in modules:
            target = f"{package}.{target}"
        if target in modules:
            imported = ", ".join(
                alias.name + (f" as {alias.asname}" if alias.asname else "")
                for alias in node.names
            )
            return f"from {modules[target]} import {imported}"
        if target in FIRST_PARTY_PACKAGES:
            statements = []
            for alias in node.names:
                child = f"{target}.{alias.name}"
                if child not in modules:
                    raise ValueError(f"Missing packaged module {child} imported by {current}")
                binding = alias.asname or alias.name
                statements.append(f"import {modules[child]} as {binding}")
            return "\n".join(statements)
        if node.level:
            raise ValueError(f"Missing relative module {target} imported by {current}")
        return None
    if isinstance(node, ast.Import):
        statements = []
        package = current.rpartition(".")[0]
        for alias in node.names:
            target = alias.name if alias.name in modules else f"{package}.{alias.name}"
            if target in modules:
                binding = alias.asname or alias.name.rsplit(".", 1)[-1]
                statements.append(f"import {modules[target]} as {binding}")
            else:
                statements.append(
                    f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                )
        return "\n".join(statements) if any(
            alias.name in modules or f"{package}.{alias.name}" in modules
            for alias in node.names
        ) else None
    return None


def _rewrite_source(path: str, modules: dict[str, str]) -> str:
    """Rewrite internal imports while preserving the original module source."""
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    encoded = source.encode("utf-8")
    edits: list[tuple[int, int, bytes]] = []
    module_name = Path(path).with_suffix("").as_posix().replace("/", ".")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        replacement = _rewrite_import(node, module_name, modules)
        if replacement is None:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        indent = " " * node.col_offset
        replacement = ("\n" + indent).join(replacement.splitlines())
        edits.append((start, end, replacement.encode("utf-8")))
    for start, end, replacement in sorted(edits, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode("utf-8")


def build_archive() -> Path:
    """Write the platform ZIP with only root-level source and pinned requirements."""
    modules = _module_map()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("solution.py", ENTRYPOINT)
        archive.writestr("requirements.txt", "\n".join(REQUIREMENTS) + "\n")
        for source_path, module_name in zip(MODULE_PATHS, modules.values()):
            archive.writestr(f"{module_name}.py", _rewrite_source(source_path, modules))
    _validate_archive(OUTPUT)
    return OUTPUT


def _validate_archive(path: Path) -> None:
    """Проверяет плоскую структуру ZIP и синтаксис всех Python-файлов."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any("/" in name or "\\" in name for name in names):
            raise ValueError("В архиве должны лежать только файлы в корне")
        if not {"solution.py", "requirements.txt"}.issubset(names):
            raise ValueError("В архиве отсутствует entrypoint или requirements.txt")
        for name in names:
            if name.endswith(".py"):
                compile(archive.read(name), name, "exec")


def main() -> None:
    """Build the ZIP and print its path and size for upload."""
    archive = build_archive()
    print(f"Created {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
