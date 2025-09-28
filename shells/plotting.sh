### Run from top level ###

# Approach 1 
python plotters/approach_1.py
python plotters/approach_1.py --lr_ablation
python plotters/approach_1.py --cooldown
python approach_1_shade_by_wd_ratio.py 

# Approach 3
python plotters/approach_3_brute_force.py --over_100 --width_depth_params
python plotters/approach_3_brute_force.py --over_100 --width_depth_params --relaxed
python plotters/approach_3_brute_force.py --over_120 --width_depth_params --relaxed
python plotters/approach_3_brute_force.py --lr_ablation --width_depth_params --relaxed
python plotters/approach_3_brute_force.py --cooldown --width_depth_params --relaxed

# alternative form
python plotters/approach_3_brute_force.py --over_100

# loss curves 
python wandb_data_plot.py

# overspending analysis
python plotters/approach_1.py --over_100 --overspending_plot

# FLOPs comparison
python plotters/approach_1.py --flops_comparison_plot

### run in plotters (i.e. cd plotters/ ) ###
# Note: most of these files have `# %%` comments which means they can be ran as notebooks in VSCode
# rainbow plot
python rainbow.py

# overtraining parabola
python overtrained_single_parabola.py

# feasible models
plot_feasible_model_shapes.ipynb

# mup analysis
python plot_mup.py

# grid search vs slope analysis
python slope_analysis.py

### run in benchmarks (i.e. cd benchmarks/) ###
# for error
python gadre_fit.py
# for accuracy
python bhagia_fit.py

python fineweb_loss_plot.py --plot