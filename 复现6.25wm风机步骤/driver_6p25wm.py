import os
from weis import weis_main

# 如果要做快速测试，可以设 True 缩短仿真时间
TEST_RUN = False

# 当前 driver.py 所在目录
run_dir = os.path.dirname(os.path.realpath(__file__))

# 你的三个输入文件路径
fname_wt_input = os.path.join(run_dir, "nrel6p25mw.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options_6p25.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_6p25.yaml")

# 调用 WEIS 主入口
wt_opt, modeling_options, opt_options = weis_main(
    fname_wt_input,
    fname_modeling_options,
    fname_analysis_options,
    test_run=TEST_RUN
)
