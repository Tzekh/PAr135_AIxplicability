import evaluate_model_2
import train_agents

#train_agents.train("/home/adam/Bureau/Transfer Learning/9 - Little Combs", "/home/adam/Bureau/3d_control_deep_rl-master/3dcdrl/scenarios_transfer_learning/little_combs_train/", 256)
#train_agents.train("/home/adam/Bureau/Transfer Learning/10 - Big Combs", "/home/adam/Bureau/3d_control_deep_rl-master/3dcdrl/scenarios_transfer_learning/big_combs_train/", 256)
#train_agents.train("/home/adam/Bureau/Transfer Learning/11 - Blend", "/home/adam/Bureau/3d_control_deep_rl-master/3dcdrl/scenarios_transfer_learning/base_comb_bigcomb/", 1024)

models = [999]

evaluate_model_2.evaluate_saved_model(models, "/home/adam/Bureau/Transfer Learning/FINAL")
