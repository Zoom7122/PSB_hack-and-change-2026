"""Собирает ZIP для платформы: solution.py и requirements.txt в корне архива."""

from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "dist" / "platform_submission.zip"
FILES = ("solution.py", "requirements.txt")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in FILES:
            archive.write(HERE / name, arcname=name)
    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        assert sorted(names) == sorted(FILES), names
        compile(archive.read("solution.py"), "solution.py", "exec")
    print(f"Создан {OUTPUT} ({OUTPUT.stat().st_size} байт): {', '.join(names)}")


if __name__ == "__main__":
    main()
