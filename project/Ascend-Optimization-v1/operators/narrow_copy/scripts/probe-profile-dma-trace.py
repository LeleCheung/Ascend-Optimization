#!/usr/bin/env python3
"""在独立 KGS debug 工作区核验新增 KG/DMA 合并候选。"""
import sys
import runpy


if __name__ == '__main__':
    sys.argv = [sys.argv[0], 'profile-small-dma-hybrid-20261010.py']
    runpy.run_path('probe-hybrid-copy-trace.py', run_name='__main__')
