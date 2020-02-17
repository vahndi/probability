from pathlib import Path
import sys

path = Path(__file__).parent
while path.name != 'probability':
    path = path.parent
sys.path.append(str(path))
