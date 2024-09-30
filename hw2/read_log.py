from tensorboard.backend.event_processing import event_accumulator
import os
import pdb


for root, dirs, files in os.walk('./data'):
    for dir in dirs:
        for file in os.listdir(f"{root}/{dir}"):
            # 文件路径
            event_file = f"{root}/{dir}/{file}"

            # 加载事件文件
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            # 获取所有标量标签
            tags = ea.Tags()['scalars']
            print(dir)
            print(ea.Scalars('Initial_DataCollection_AverageReturn')[0].value)
            # pdb.set_trace()
            # # 打印每个标签的标量值
            # for tag in tags:
            #     scalars = ea.Scalars(tag)
            #     # pdb.set_trace()
            #     # for scalar in scalars:
            #     #     print(f"Step: {scalar.step}, Wall time: {scalar.wall_time}, Value: {scalar.value}")
            #     print(f"{tag}:{len(scalars)}")