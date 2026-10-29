"""Печать метаданных запуска: python tools/run_meta.py"""
import json

from penza_yield.meta import collect

print(json.dumps(collect(), ensure_ascii=False, indent=2))
