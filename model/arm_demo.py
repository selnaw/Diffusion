import mani_skill.envs
import gymnasium as gym

# 关键：render_mode="human" 就是弹出可视化窗口
env = gym.make("PushCube-v1", render_mode="human")

# 初始化场景
obs, _ = env.reset()

# 运行1000步，机械臂会持续动起来，窗口也会正常刷新
for _ in range(1000):
    # 让机械臂执行动作
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # 刷新窗口画面（必须写这行，不然窗口会卡住）
    env.render()
    # 如果游戏结束就重置场景
    if terminated or truncated:
        obs, _ = env.reset()

# 运行完关闭环境
env.close()