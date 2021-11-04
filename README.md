# MADDPG_based_on_Pytorch
# This object is based on Pytorch , and I refer to the PARL.
'''
    由于对于多智能体强化学习的需要，用pytorch写了这一篇基于MPE环境的MADDPG算法。
    之前做毕业设计时候，作者主要通过MADDPG论文附带的代码以及百度的PARL库了解MADDPG，论文中附带的代码
基于Tensorflow, PARL则运用了自家的paddle库，因此作者很久之前就想要再写一份基于torch的MADDPG, 刚好借
着这次深入了解MADDPG算法的契机，作者学习先辈们的经验，创作了这份代码。
    注:本项目主要参考PARL(其中有小部分函数实现使用了PARL以及paddle的方法),
    PARL链接:https://github.com/PaddlePaddle/PARL.git
'''
# 项目结构
'''
    项目主要分为四部分:
    model.py # 模型
    alg.py  # MADDPG算法
    agent.py  # 智能体
    train.py  # 训练主函数
    para.py  # 参数
    replay_memory.py  #  经验池
    function.py  # 一些其他函数
    difference in torch and paddle with parl  # 作者的一些总结
 '''
 
 # python环境
 '''
    gym==0.10.5
    numpy==1.19.2
    torch==1.7.1+cu110
    https://github.com/openai/multiagent-particle-envs
    python==3.7
  '''
  
  # 本项目属于学习用，如果有错误或不严密的地方，欢迎大家留言指正
    
