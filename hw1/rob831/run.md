# HW1 Commands

## 1 Behavioral Cloning

### 1.2

```shell
python rob831/scripts/q1_expert_eval.py
```

### 1.3

`Ant-v2`
```shell
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1 \
--do_dagger False \
--batch_size 1000 \
--train_batch_size 100 \
--num_agent_train_steps_per_iter 1000 \
--eval_batch_size 5000 \
--n_layers 2 \
--size 64 \
--learning_rate 5e-3 \
--seed 1
```

`Humanoid-v2`
```shell
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 \
--do_dagger False \
--batch_size 1000 \
--train_batch_size 100 \
--num_agent_train_steps_per_iter 1000 \
--eval_batch_size 5000 \
--n_layers 2 \
--size 64 \
--learning_rate 5e-3 \
--seed 1
```

### 1.4

Change 1.3 with `--num_agent_train_steps_per_iter` in `[500, 1000, 2000, 4000]`  

Snippet used for plotting:
```python

def plot_bc_hyperparameter_experiment():
    hyperparam_values = [500, 1000, 2000, 4000]
    means = [2492.25, 4631.60, 4567.40, 4235.32] 
    stds = [1420.77, 115.64, 86.94, 1130.77]  
    expert_mean = 4713.6
    expert_std = 12.2

    plt.figure(figsize=(10, 6))
    
    plt.errorbar(hyperparam_values, means, yerr=stds, 
                 fmt='-o', capsize=5, capthick=2, elinewidth=2, 
                 label='BC Agent Performance', color='blue')
    
    plt.axhline(y=expert_mean, color='red', linestyle='--', label='Expert Performance')
    plt.fill_between([min(hyperparam_values), max(hyperparam_values)], 
                     expert_mean - expert_std, expert_mean + expert_std, 
                     color='red', alpha=0.1)

    plt.xlabel('Number of Training Steps per Iteration', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('Effect of Training Steps on BC Agent Performance (Ant-v2)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(hyperparam_values)
    
    plt.tight_layout()
    plt.show()

plot_bc_hyperparameter_experiment()
```

## 2 DAgger

### 2.1

`Ant-v2`
```shell
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1 \
--do_dagger True \
--batch_size 1000 \
--train_batch_size 100 \
--num_agent_train_steps_per_iter 1000 \
--eval_batch_size 5000 \
--n_layers 2 \
--size 64 \
--learning_rate 5e-3 \
--seed 1
```

`Humanoid-v2`
```shell
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 10 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 \
--do_dagger True \
--batch_size 1000 \
--train_batch_size 100 \
--num_agent_train_steps_per_iter 1000 \
--eval_batch_size 5000 \
--n_layers 2 \
--size 64 \
--learning_rate 5e-3 \
--seed 1
```

Snippet used for plotting:
```python
def plot_dagger_results():
    iters_ant = list(range(10)) 
    means_ant = [
        4631.61, 4579.98, 4654.82, 4620.61, 4496.94, 
        4363.40, 4800.61, 4678.53, 4736.69, 4727.48
    ]
    stds_ant = [
        115.64, 145.82, 100.63, 75.98, 465.60, 
        748.91, 93.69, 43.91, 67.61, 58.16
    ]
    expert_mean_ant = 4713.65 
    
    iters_humanoid = list(range(10))
    means_humanoid = [
        305.37, 276.47, 272.62, 313.61, 334.93,
        319.98, 324.58, 341.34, 366.77, 352.43
    ]
    stds_humanoid = [
        59.92, 25.94, 44.65, 33.53, 69.28,
        38.88, 37.83, 38.57, 68.29, 59.52
    ]
    expert_mean_humanoid = 10344.52

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.errorbar(iters_ant, means_ant, yerr=stds_ant, fmt='-o', color='blue', label='DAgger Agent')
    ax1.axhline(y=expert_mean_ant, color='green', linestyle='--', label='Expert Performance')
    ax1.axhline(y=means_ant[0], color='red', linestyle=':', label='BC Performance')
    
    ax1.set_title('Ant-v2: DAgger Performance')
    ax1.set_xlabel('DAgger Iterations')
    ax1.set_ylabel('Mean Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(iters_humanoid, means_humanoid, yerr=stds_humanoid, fmt='-o', color='orange', label='DAgger Agent')
    ax2.axhline(y=expert_mean_humanoid, color='green', linestyle='--', label='Expert Performance')
    ax2.axhline(y=means_humanoid[0], color='red', linestyle=':', label='BC Performance')
    
    ax2.set_title('Humanoid-v2: DAgger Performance')
    ax2.set_xlabel('DAgger Iterations')
    ax2.set_ylabel('Mean Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_dagger_results()
```
