"""Build a flat platform ZIP by following the actual source import graph."""

import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist/psb_hack_change_submission.zip"
ENTRY_MODULES = ("solution", "answer_generation.verify")
PACKAGES = ("API", "answer_generation", "data_processing", "query_processing_module", "rag_pipeline")
REQUIREMENTS = (
    "httpx==0.28.1", "numpy==2.2.6", "openai==3.45.0",
    "pydantic==2.14.2", "python-dotenv==1.1.1", "rank-bm25==0.2.2",
)


def _source_path(module: str) -> Path:
    """Resolve an original module name inside the project source tree."""
    return ROOT.joinpath(*module.split(".")).with_suffix(".py")


def _relative_name(current: str, level: int, module: str | None) -> str:
    """Resolve a package-relative import without executing source modules."""
    package = current.split(".")[:-1]
    if level > len(package):
        raise ValueError(f"Relative import escapes package: {current}")
    prefix = package[:len(package) - level + 1]
    return ".".join(prefix + (module.split(".") if module else []))


def _local_target(target: str, current: str) -> str | None:
    """Find an exact local module or a legacy sibling-module import."""
    if _source_path(target).is_file():
        return target
    sibling = current.rpartition(".")[0] + "." + target
    return sibling if _source_path(sibling).is_file() else None


def _from_target(node: ast.ImportFrom, current: str) -> str:
    """Resolve the left side of an import-from statement."""
    return _relative_name(current, node.level, node.module) if node.level else node.module or ""


def _dependencies(node: ast.AST, current: str) -> list[str]:
    """Collect internal dependencies and reject missing first-party imports."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        target = _from_target(node, current)
        exact = _local_target(target, current)
        if exact:
            return [exact]
        names = [target + "." + alias.name for alias in node.names]
    else:
        return []
    result = []
    for name in names:
        target = _local_target(name, current)
        if target:
            result.append(target)
        elif name.split(".")[0] in PACKAGES:
            raise ValueError(f"Missing local module {name} imported by {current}")
    if isinstance(node, ast.ImportFrom) and node.level and not result:
        raise ValueError(f"Unresolved relative import in {current}")
    return result


def discover_modules() -> dict[str, str]:
    """Follow imports from both CLI entry points, excluding tests and fixtures."""
    pending, modules = list(ENTRY_MODULES), {}
    while pending:
        current = pending.pop(0)
        if current in modules:
            continue
        source = _source_path(current).read_text(encoding="utf-8")
        modules[current] = "solution" if current == "solution" else "_psb_" + current.replace(".", "_").lower()
        for node in ast.walk(ast.parse(source, filename=current)):
            pending.extend(_dependencies(node, current))
    if len(set(modules.values())) != len(modules):
        raise ValueError("Flattened module names collide")
    return modules


def _rewrite_import(node: ast.AST, current: str, modules: dict[str, str]) -> str | None:
    """Rewrite package and legacy imports into names of root-level files."""
    if isinstance(node, ast.ImportFrom):
        target = _from_target(node, current)
        exact = _local_target(target, current)
        if exact in modules:
            names = ", ".join(alias.name + (f" as {alias.asname}" if alias.asname else "")
                              for alias in node.names)
            return f"from {modules[exact]} import {names}"
        children = [_local_target(target + "." + alias.name, current) for alias in node.names]
        if any(children):
            if not all(child in modules for child in children):
                raise ValueError(f"Mixed package and symbol import in {current}")
            return "\n".join(f"import {modules[child]} as {alias.asname or alias.name}"
                             for child, alias in zip(children, node.names))
    if isinstance(node, ast.Import):
        statements, changed = [], False
        for alias in node.names:
            target = _local_target(alias.name, current)
            if target in modules:
                if "." in alias.name and not alias.asname:
                    raise ValueError(f"Use an alias for dotted module import {alias.name} in {current}")
                statements.append(f"import {modules[target]} as {alias.asname or alias.name}")
                changed = True
            else:
                statements.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        return "\n".join(statements) if changed else None
    return None


def _rewrite_source(current: str, modules: dict[str, str]) -> str:
    """Change only import spans, leaving module logic and docstrings intact."""
    source = _source_path(current).read_text(encoding="utf-8")
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    edits = []
    for node in ast.walk(ast.parse(source, filename=current)):
        replacement = _rewrite_import(node, current, modules)
        if replacement is None:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        replacement = ("\n" + " " * node.col_offset).join(replacement.splitlines())
        edits.append((start, end, replacement.encode("utf-8")))
    encoded = source.encode("utf-8")
    for start, end, replacement in sorted(edits, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode("utf-8")


def _validate_archive(path: Path) -> None:
    """Check archive integrity, allowed members, required entry point and syntax."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if archive.testzip() or len(names) != len(set(names)):
            raise ValueError("Corrupt ZIP or duplicate archive entries")
        if any("/" in name or "\\" in name or not (name.endswith(".py") or name == "requirements.txt") for name in names):
            raise ValueError("Platform ZIP must contain only root Python files and requirements.txt")
        if not {"solution.py", "requirements.txt"}.issubset(names):
            raise ValueError("Missing required platform entry point")
        for name in names:
            if name.endswith(".py"):
                compile(archive.read(name), name, "exec")
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Platform ZIP exceeds 128 MiB")


def build_archive(output: Path = OUTPUT) -> Path:
    """Build a reproducible ZIP and write a separate source-hash manifest."""
    modules = discover_modules()
    files = {name + ".py": _rewrite_source(module, modules) for module, name in modules.items()}
    files["requirements.txt"] = "\n".join(REQUIREMENTS) + "\n"
    files["verify_submission.py"] = (
        '"""Recheck saved answers using the same verifier as solution.py."""\n'
        "from _psb_answer_generation_verify import main\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
    _validate_archive(output)
    manifest = {"archive": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "modules": {module: {"file": name + ".py",
                            "source_sha256": hashlib.sha256(_source_path(module).read_bytes()).hexdigest()}
                            for module, name in sorted(modules.items())}}
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    """Build the upload artifact using the current project modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = build_archive(parser.parse_args().output)
    print(f"Created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
