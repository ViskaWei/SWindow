import sys
import os
import numpy as np
import pandas as pd
# import getpass
# print(os.getcwd())
# sys.path.insert(0,'/home/swei20/cancerHH/AceCanZ/code/')
# from pipeline.cmdPipeline import CmdPipeline
from pipeline.symNormPipeline import SymNormPipeline

def main():
    p=SymNormPipeline()
    p.prepare()
    p.run()

    # print(p.args)
if __name__ == "__main__":
    main()
