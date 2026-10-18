import sys
from pathlib import Path

file_path = Path(__file__).resolve()
while file_path.name != 'probability':
    file_path = file_path.parent
if str(file_path) not in sys.path:
    sys.path.append(str(file_path))
