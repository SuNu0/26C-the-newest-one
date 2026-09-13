2026年全国大学生数学建模竞赛 C 题支撑材料

目录：code 为求解与验证程序；附件为运行输入及结果模板；results 为主要结果；figures 为程序运行时的图片输出目录，初始为空。

建议使用 Python 3.11 或更高版本。
建议使用 UTF-8 终端。在本文件所在目录执行。

安装依赖：
python -m pip install -r requirements-problem1.txt
python -m pip install -r requirements-q234.txt

快速验证（问题一通常约10秒，问题二至四通常约1分钟，以实际机器为准）：
python code/problem1_solve.py --smoke
python code/q234_solve.py --smoke
python code/test_time_alignment.py

完整重算（可能需要数分钟）：
python code/problem1_solve.py
python code/problem1_validate.py
python code/q234_solve.py
python code/q234_validate.py
python code/test_time_alignment.py

上述求解和验证程序会在 results 和 figures 中生成或覆盖同名结果。赛题原始附件一至附件四随包保留，用于保证程序可独立运行。完整逐时段结果已放入 results，也可由 q234_solve.py 重新生成。
