import sys
import os
import numpy as np
import pandas as pd
import argparse

from symnorm.pipeline.symNormPipeline import SymNormPipeline

def main():
    # sys.argv = ["main", "--config", "./configs/testConfigs.json"]
    # sys.argv = ["main", "--config", "./configs/csLConfigs.json"]
    sys.argv = ["main", "--config", "./configs/mLConfigs.json"]

    p=SymNormPipeline()
    p.prepare()
    p.run()

    # print(p.args)
if __name__ == "__main__":
    main()
