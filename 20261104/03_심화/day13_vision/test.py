import os
from pathlib import Path

print(os.getcwd())

path = Path(__file__).resolve().parent

with open(path / "test.txt","r",encoding="utf-8") as f:
    text = f.read()

names = text.split(",")
print(names)