#!/usr/bin/env python3
"""Process every supported pure-species trajectory in a directory."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from waterpost.collection import main
if __name__=='__main__':sys.exit(main())
