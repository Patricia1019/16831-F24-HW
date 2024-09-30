import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pdb

# HalfCheetah search

# set parameters 7.2.2

all_returns = {}

for root, dirs, files in os.walk('./data'):
    for dir in dirs:
        if f'q4_search_b10000_lr0.02' in dir:
            for file in os.listdir(f"{root}/{dir}"):
                event_file = f"{root}/{dir}/{file}"
                
                # 处理 TensorFlow 事件文件
                try:
                    ea = event_accumulator.EventAccumulator(event_file)
                    ea.Reload()

                    # 如果事件文件包含 'Eval_AverageReturn' 标量
                    if 'Eval_AverageReturn' in ea.Tags()['scalars']:
                        eval_returns = ea.Scalars('Eval_AverageReturn')
                        
                        # 提取 step 和 return 值
                        steps = [scalar.step for scalar in eval_returns]
                        values = [scalar.value for scalar in eval_returns]
                        
                        # 保存每个文件的 'Eval_AverageReturn' 值
                        all_returns[dir] = (steps, values)
                except Exception as e:
                    print(f"Error processing {event_file}: {e}")

# 绘制所有文件的 'Eval_AverageReturn' 值
plt.figure(figsize=(10, 6))
for file_name, (steps, values) in all_returns.items():
    label = file_name.split('lr0.02_')[1]
    label = label.split('_HalfCheetah')[0]
    if 'HalfCheetah' in label:
        label = "none"
    plt.plot(steps, values, label=label)

# 添加图例和标签
plt.title('Eval_AverageReturn for b10000_lr0.02')
plt.xlabel('Steps')
plt.ylabel('Eval_AverageReturn')
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# 展示图像
plt.savefig('./images/7.2.2.png')



# q4 search

all_returns = {}

for root, dirs, files in os.walk('./data'):
    for dir in dirs:
        if f'q4_search' in dir and 'rtg_nnbaseline' in dir:
            for file in os.listdir(f"{root}/{dir}"):
                event_file = f"{root}/{dir}/{file}"
                
                # 处理 TensorFlow 事件文件
                try:
                    ea = event_accumulator.EventAccumulator(event_file)
                    ea.Reload()

                    # 如果事件文件包含 'Eval_AverageReturn' 标量
                    if 'Eval_AverageReturn' in ea.Tags()['scalars']:
                        eval_returns = ea.Scalars('Eval_AverageReturn')
                        
                        # 提取 step 和 return 值
                        steps = [scalar.step for scalar in eval_returns]
                        values = [scalar.value for scalar in eval_returns]
                        
                        # 保存每个文件的 'Eval_AverageReturn' 值
                        all_returns[dir] = (steps, values)
                except Exception as e:
                    print(f"Error processing {event_file}: {e}")

# 绘制所有文件的 'Eval_AverageReturn' 值
plt.figure(figsize=(10, 6))
for file_name, (steps, values) in all_returns.items():
    label = file_name.split('_rtg')[0]
    label = label.split('q4_search_')[1]
    plt.plot(steps, values, label=label)

# 添加图例和标签
plt.title('Eval_AverageReturn for q4 Search')
plt.xlabel('Steps')
plt.ylabel('Eval_AverageReturn')
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# 展示图像
plt.savefig('./images/7.2.4.png')


# best b and best lr four runs

all_returns = {}

for root, dirs, files in os.walk('./data'):
    for dir in dirs:
        if f'q4_b50000_lr0.02' in dir:
            for file in os.listdir(f"{root}/{dir}"):
                event_file = f"{root}/{dir}/{file}"
                
                # 处理 TensorFlow 事件文件
                try:
                    ea = event_accumulator.EventAccumulator(event_file)
                    ea.Reload()

                    # 如果事件文件包含 'Eval_AverageReturn' 标量
                    if 'Eval_AverageReturn' in ea.Tags()['scalars']:
                        eval_returns = ea.Scalars('Eval_AverageReturn')
                        
                        # 提取 step 和 return 值
                        steps = [scalar.step for scalar in eval_returns]
                        values = [scalar.value for scalar in eval_returns]
                        
                        # 保存每个文件的 'Eval_AverageReturn' 值
                        all_returns[dir] = (steps, values)
                except Exception as e:
                    print(f"Error processing {event_file}: {e}")

# 绘制所有文件的 'Eval_AverageReturn' 值
plt.figure(figsize=(10, 6))
for file_name, (steps, values) in all_returns.items():
    label = file_name.split('lr0.02_')[1]
    label = label.split('_HalfCheetah')[0]
    if 'HalfCheetah' in label:
        label = "none"
    plt.plot(steps, values, label=label)

# 添加图例和标签
plt.title('Eval_AverageReturn for four runs(b50000_lr0.02)')
plt.xlabel('Steps')
plt.ylabel('Eval_AverageReturn')
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# 展示图像
plt.savefig('./images/7.2.7.png')

