# --num_parameters 2 => parameters, tokens laws
# --num_parameters 3 => width, depth, tokens laws
# --num_parameters 4 => width, depth, params, tokens laws

SAVEDIR=parameters
mkdir -p $SAVEDIR

# all data
python depth_width.py --num_parameters 2 --hot_over_100 --num_processes 96 --save_path $SAVEDIR
python depth_width.py --num_parameters 3 --hot_over_100 --num_processes 96 --save_path $SAVEDIR
python depth_width.py --num_parameters 4 --hot_over_100 --num_processes 96 --save_path $SAVEDIR

# <= 100b data
python depth_width.py --num_parameters 2 --hot --num_processes 96 --save_path $SAVEDIR
python depth_width.py --num_parameters 4 --hot --num_processes 96 --save_path $SAVEDIR

# lr/2 runs
python depth_width.py --num_parameters 2 --lr_ablation_hot --num_processes 96 --save_path $SAVEDIR
python depth_width.py --num_parameters 4 --lr_ablation_hot --num_processes 96 --save_path $SAVEDIR

# cooldown runs
python depth_width.py --num_parameters 2 --cool_end --num_processes 96 --save_path $SAVEDIR
python depth_width.py --num_parameters 4 --cool_end --num_processes 96 --save_path $SAVEDIR

# > 120b data
python depth_width.py --num_parameters 2 --hot_over_120 --num_processes 96 --save_path $SAVEDIR
python depth_width.py --num_parameters 4 --hot_over_120 --num_processes 96 --save_path $SAVEDIR

# 512x models only
python depth_width.py --num_parameters 2 --hot_over_100_512x --num_processes 96 --save_path $SAVEDIR

# filtered to be like chinchilla
python depth_width.py --num_parameters 2 --like_chinchilla --num_processes 96 --save_path $SAVEDIR # our data
python depth_width.py --num_parameters 2 --like_chinchilla_lr_ablation --num_processes 96 --save_path $SAVEDIR # our data
python depth_width.py --num_parameters 2 --slim_chinchilla --num_processes 96 --save_path $SAVEDIR # Chinchilla data
python depth_width.py --num_parameters 2 --lr_ablation_slim_chinchilla --num_processes 96 --save_path $SAVEDIR # Chinchilla data