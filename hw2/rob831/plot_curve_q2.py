import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pdb

# InvertedPendulum

# 用于保存所有文件的步骤和 'Eval_AverageReturn' 的值
all_returns = {}

# 遍历指定目录下的所有文件
for root, dirs, files in os.walk('./data'):
    for dir in dirs:
        if "InvertedPendulum" in dir:
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
    label = file_name.split('_InvertedPendulum')[0]
    plt.plot(steps, values, label=label)

# 添加图例和标签
plt.title('Eval_AverageReturn for Inverted Pendulum Comparison')
plt.xlabel('Steps')
plt.ylabel('Eval_AverageReturn')
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# 展示图像
plt.savefig('./images/5.2.3_compare.png')
plt.close()


# best b and best r
all_returns = {}

best_b = 50
best_r = 0.02
for root, dirs, files in os.walk('./data'):
    for dir in dirs:
        if f'b{best_b}_r{best_r}' in dir:
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
    label = file_name.split('_InvertedPendulum')[0]
    plt.plot(steps, values, label=label)

# 添加图例和标签
plt.title('Eval_AverageReturn for Best Params for Inverted Pendulum')
plt.xlabel('Steps')
plt.ylabel('Eval_AverageReturn')
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# 展示图像
plt.savefig('./images/5.2.3.png')


