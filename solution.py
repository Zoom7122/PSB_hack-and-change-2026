"""Точка входа платформенного CSV-запуска решения."""

from API.for_solutions import main
import sys


if __name__ == "__main__":
    main(sys.argv[1:])
