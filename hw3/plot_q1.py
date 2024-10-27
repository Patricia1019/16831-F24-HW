import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pdb

STEP_LEN = 30
STEPS = []
# 用于保存所有文件的步骤和 'Eval_AverageReturn' 的值
dqn_sum = [0]*STEP_LEN
dqn_best = {}
for i in range(10001,300002,10000):
    dqn_best[i] = 0


# 遍历指定目录下的所有文件
for root, dirs, files in os.walk('./data'):
    for dir in dirs:
        if "q1_dqn" in dir:
            for file in os.listdir(f"{root}/{dir}"):
                event_file = f"{root}/{dir}/{file}"
                
                # 处理 TensorFlow 事件文件
                # try:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                # 如果事件文件包含 'Eval_AverageReturn' 标量
                if 'Train_AverageReturn' in ea.Tags()['scalars'] and 'Train_BestReturn' in ea.Tags()['scalars']:
                    eval_returns = ea.Scalars('Train_AverageReturn')
                    
                    # 提取 step 和 return 值
                    steps = [scalar.step for scalar in eval_returns]
                    STEPS = steps
                    values = [scalar.value for scalar in eval_returns]
                    
                    # 保存每个文件的 'Eval_AverageReturn' 值
                    dqn_sum = [dqn_sum[i]+values[i] for i in range(STEP_LEN)]

                    # best return
                    eval_returns = ea.Scalars('Train_BestReturn')
                    
                    # 提取 step 和 return 值
                    steps = [scalar.step for scalar in eval_returns]
                    values = [scalar.value for scalar in eval_returns]
        
                    # 保存每个文件的 'Eval_AverageReturn' 值
                    for i in range(len(steps)):
                        dqn_best[steps[i]] += values[i]
                # except Exception as e:
                #     print(f"Error processing {event_file}: {e}")
dqn_avg = [i/3 for i in dqn_sum]
for i in dqn_best.copy().keys():
    dqn_best[i] /= 3
    if dqn_best[i] == 0:
        del dqn_best[i]

# 用于保存所有文件的步骤和 'Eval_AverageReturn' 的值
ddqn_sum = [0]*STEP_LEN
ddqn_best = {}
for i in range(10001,300002,10000):
    ddqn_best[i] = 0

# 遍历指定目录下的所有文件
for root, dirs, files in os.walk('./data'):
    for dir in dirs:
        if "q1_doubledqn" in dir:
            for file in os.listdir(f"{root}/{dir}"):
                event_file = f"{root}/{dir}/{file}"
                
                # 处理 TensorFlow 事件文件
                try:
                    ea = event_accumulator.EventAccumulator(event_file)
                    ea.Reload()
                    # pdb.set_trace()
                    # 如果事件文件包含 'Eval_AverageReturn' 标量
                    if 'Train_AverageReturn' in ea.Tags()['scalars'] and 'Train_BestReturn' in ea.Tags()['scalars']:
                        eval_returns = ea.Scalars('Train_AverageReturn')
                        
                        # 提取 step 和 return 值
                        steps = [scalar.step for scalar in eval_returns]
                        values = [scalar.value for scalar in eval_returns]
                        
                        # 保存每个文件的 'Eval_AverageReturn' 值
                        ddqn_sum = [ddqn_sum[i]+values[i] for i in range(STEP_LEN)]

                        # best return
                        eval_returns = ea.Scalars('Train_BestReturn')
                        
                        # 提取 step 和 return 值
                        steps = [scalar.step for scalar in eval_returns]
                        values = [scalar.value for scalar in eval_returns]
                        
                        # 保存每个文件的 'Eval_AverageReturn' 值
                        for i in range(len(steps)):
                            ddqn_best[steps[i]] += values[i]
                except Exception as e:
                    print(f"Error processing {event_file}: {e}")
ddqn_avg = [i/3 for i in ddqn_sum]
for i in ddqn_best.copy().keys():
    ddqn_best[i] /= 3
    if ddqn_best[i] == 0:
        del ddqn_best[i]

# 绘制所有文件的 'Eval_AverageReturn' 值
plt.figure(figsize=(10, 6))
# for file_name, (steps, values) in ddqn_returns.items():
#     label = file_name.split('q1_')[0]
plt.plot(STEPS, dqn_avg, label="dqn_average")
plt.plot(STEPS, ddqn_avg, label="doubledqn_average")
plt.plot(dqn_best.keys(), dqn_best.values(), label="dqn_best")
plt.plot(ddqn_best.keys(), ddqn_best.values(), label="doubledqn_best")

# 添加图例和标签
# plt.title('Eval_AverageReturn for Small Batch Experiments')
plt.xlabel('Number of Time Steps')
plt.ylabel('Return')
plt.legend(loc='lower right', fontsize='large')
plt.grid(True)

# 展示图像
plt.savefig('./images/q1.png')
