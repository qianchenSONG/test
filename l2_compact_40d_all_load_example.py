"""读取 l2_compact_40d_all.npz 的示例。

文件结构：
  '<日期>/<代码>/<字段>'   每债每日的原始数组，字段、单位、dtype 与原来的单个 npz 完全相同
                            （O H L C V AMT BA SA NT qt last ap bp av bv cumamt tav tbv pre sh）
  '_index/date' 'code' 'sh' 'pre' 'first_day' 'n_snap'   债日索引（12,344 行，按日期、代码排序）
  '_index/fields'           字段列表
  '_meta/daily_csv' 'overview_csv' 'readme'   日K汇总、每日概览、README 的原文

np.load 是按需读取的：只有取到的数组才会解压，整个文件不会一次性读进内存。
"""
import io
import sys
from pathlib import Path
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else 'l2_compact_40d_all.npz'
z = np.load(PATH)
FIELDS = list(z['_index/fields'])


def load(date, code):
    """取一个债日，返回和原来 np.load('<日期>/<代码>.npz') 一样的字典。"""
    return {k: z[f'{date}/{code}/{k}'] for k in FIELDS}


def index_table():
    """债日索引，返回 pandas DataFrame（需要 pandas）。"""
    import pandas as pd
    return pd.DataFrame({k: z[f'_index/{k}'] for k in
                         ['date', 'code', 'sh', 'pre', 'first_day', 'n_snap']})


def daily_k():
    """日K汇总_每债每日.csv 的内容（需要 pandas）。"""
    import pandas as pd
    return pd.read_csv(io.StringIO(str(z['_meta/daily_csv'])), dtype={'code': str})


def extract_all(out_dir):
    """还原成原来的目录结构 <out_dir>/<日期>/<代码>.npz，旧脚本可直接读取。"""
    out = Path(out_dir)
    for d, c in zip(z['_index/date'], z['_index/code']):
        (out / str(d)).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / str(d) / f'{c}.npz', **load(d, c))
    for name, key in [('日K汇总_每债每日.csv', 'daily_csv'), ('每日概览.csv', 'overview_csv'),
                      ('README.md', 'readme')]:
        (out / name).write_text(str(z[f'_meta/{key}']), encoding='utf-8')


if __name__ == '__main__':
    date, code = z['_index/date'], z['_index/code']
    print('债日数', len(date), '交易日', len(np.unique(date)), '债数', len(np.unique(code)))
    print('上市首日（回测剔除）', int(z['_index/first_day'].sum()), '个债日')

    b = load(20260929, '113052')
    print('113052 @0929 前收', b['pre'], '收盘', b['C'][-1], '快照数', len(b['qt']))

    # 取某一天全部债（剔除上市首日）
    sel = (date == 20260930) & ~z['_index/first_day']
    day = {c: load(20260930, c) for c in code[sel]}
    print('0930 债数', len(day))

    # 取某只债全部日期的分钟收盘价
    m = code == '113052'
    closes = np.stack([z[f'{d}/113052/C'] for d in date[m]])
    print('113052 分钟收盘价矩阵', closes.shape)
