import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from AFM_Github.forward_prediction.prediction import MyModel, DataLoader, predict, MyDataSet
import torch
from pathlib import Path
from sko.PSO import PSO
from functools import partial
from scipy.interpolate import interp1d

num_features = 8
num_output_dims = 21
device = 'cuda'

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
path_all_NN = project_root / 'forward_prediction'


def deep_learn_predict(input, model):
    """
    :param input: the structure parameters, num_samples * features (ndarray)
    :param model: the ANN model used to predict the mechanical response
    :return: the predict mechanical response
    """
    # fix batch size to  all data_x
    num_samples = input.shape[0]
    dataset_input = MyDataSet(input)
    input_loader = DataLoader(dataset_input, batch_size=num_samples, shuffle=False, pin_memory=True)
    preds = predict(input_loader, model, device)
    return preds


def load_pred_model(save_path):
    # load the trained model
    model_pred = MyModel(in_dim=num_features, out_dim=num_output_dims).to(device)
    model_pred.load_state_dict(torch.load(save_path, weights_only=True))
    return model_pred


def my_mse_fun(pred_res, target_curve):
    # pred_res: 1*20  strain range: 0-3
    # target: n*2; strain range:

    pred_x = np.linspace(0, 0.3, 21)
    interpolation_function = interp1d(pred_x, pred_res[0], kind='cubic')
    x_target = target_curve[:, 0]
    pred_new = interpolation_function(x_target)
    y_target = target_curve[:, 1]
    return np.sum(np.square(pred_new - y_target)) / target_curve.shape[1]


def pso_design_displace_target_function(samples, model_target_pairs_list):
    samples = samples.reshape(-1, num_features)
    all_mse = []
    for model_target_pair in model_target_pairs_list:
        model = model_target_pair.model
        target_curve = model_target_pair.target
        pred_res_displace = deep_learn_predict(samples, model)
        # pred_res_z = deep_learn_predict(samples, model_z)
        all_mse.append(my_mse_fun(pred_res_displace, target_curve))
    return sum(all_mse)


def main_pso_optimize(model_target_pairs_list, inverse_design_result_path):
    pso_design_displace_target_function_fix_para = partial(pso_design_displace_target_function,
                                                           model_target_pairs_list=model_target_pairs_list)
    num_feature_solve = num_features
    low_bounds = [0.] * num_feature_solve
    up_bounds = [1.] * num_feature_solve
    pso = PSO(func=pso_design_displace_target_function_fix_para, n_dim=num_feature_solve, pop=128, max_iter=32,
              lb=low_bounds, ub=up_bounds, w=0.8, c1=0.5, c2=0.5)
    pso.run()

    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    # create the inverse design result storage folder
    # save the best result
    np.savetxt(os.path.join(inverse_design_result_path, 'feature.txt'), pso.gbest_x)
    np.savetxt(os.path.join(inverse_design_result_path, 'target_function_history.txt'), pso.gbest_y_hist)
    np.savetxt(os.path.join(inverse_design_result_path, 'target_function_best.txt'), pso.gbest_y)

    plt.plot(pso.gbest_y_hist)
    plt.show()
    return pso.gbest_x


def plot_res_best(best_x, model_target_pairs_list, inverse_design_result_path):
    fig_res = plt.figure('response')
    ax_res = fig_res.subplots()
    num_targets = len(model_target_pairs_list)
    for i in range(num_targets):
        model_target_pair = model_target_pairs_list[i]
        model = model_target_pair.model
        target_curve = model_target_pair.target
        pred = deep_learn_predict(best_x.reshape(1, -1), model)
        ax_res.plot(np.linspace(0.0, 0.3, 21), pred[0], '-')
        ax_res.plot(target_curve[:, 0], target_curve[:, 1], 'o')
    plt.show()


def get_pred_model(model_para_path):
    model_pred_displace = load_pred_model(model_para_path)
    return model_pred_displace


class ModelTargetPair:
    def __init__(self, model, target_curve):
        self.model = model
        self.target = target_curve


def main():
    # 读取的数据为应力，而神经网络预测的是反力，因此不太匹配
    stress_z_muscle_original = np.loadtxt('./data/skeletal muscle z.txt')
    stress_y_muscle_original = np.loadtxt('./data/skeletal muscle y.txt')
    stress_z_muscle_biaxial_original = np.loadtxt('./data/skeletal muscle biaxial z.txt')
    stress_y_muscle_biaxial_original = np.loadtxt('./data/skeletal muscle biaxial y.txt')

    # remover the rows which the excess the design range
    stress_z_muscle = stress_z_muscle_original[(stress_z_muscle_original[:, 0] < 0.3) & (stress_z_muscle_original[:, 0] > 0.0), :]
    stress_y_muscle = stress_y_muscle_original[(stress_y_muscle_original[:, 0] < 0.3) & (stress_y_muscle_original[:, 0] > 0.0), :]
    stress_z_muscle_biaxial = stress_z_muscle_biaxial_original[(stress_z_muscle_biaxial_original[:, 0] < 0.3) & (stress_z_muscle_biaxial_original[:, 0] > 0.0), :]
    stress_y_muscle_biaxial = stress_y_muscle_biaxial_original[(stress_y_muscle_biaxial_original[:, 0] < 0.3) & (stress_y_muscle_biaxial_original[:, 0] > 0.0), :]

    np.random.seed(112)
    target_curve_z = stress_z_muscle
    target_curve_y = stress_y_muscle
    model_para_path_z = os.path.join(path_all_NN, 'model_uniaxial_z/model.ckpt')
    model_para_path_y = os.path.join(path_all_NN, 'model_uniaxial_y/model.ckpt')
    model_para_path_biaxial_z = os.path.join(path_all_NN, 'model_biaxial_z/model.ckpt')
    model_para_path_biaxial_y = os.path.join(path_all_NN, 'model_biaxial_y/model.ckpt')

    model_z = get_pred_model(model_para_path_z)
    model_y = get_pred_model(model_para_path_y)
    model_biaxial_z = get_pred_model(model_para_path_biaxial_z)
    model_biaxial_y = get_pred_model(model_para_path_biaxial_y)
    model_target_pair_z = ModelTargetPair(model_z, target_curve_z)
    model_target_pair_y = ModelTargetPair(model_y, target_curve_y)
    model_target_pair_biaxial_z = ModelTargetPair(model_biaxial_z, stress_z_muscle_biaxial)
    model_target_pair_biaxial_y = ModelTargetPair(model_biaxial_y, stress_y_muscle_biaxial)
    model_target_pairs_list = [model_target_pair_z, model_target_pair_y, model_target_pair_biaxial_z,
                               model_target_pair_biaxial_y]
    inverse_design_result_path = './res_muscle'
    os.makedirs(inverse_design_result_path, exist_ok=True)
    best_sample = main_pso_optimize(model_target_pairs_list, inverse_design_result_path)
    # best_sample = np.loadtxt(os.path.join(inverse_design_result_path, 'feature.txt'))
    plot_res_best(best_sample, model_target_pairs_list=model_target_pairs_list,
                  inverse_design_result_path=inverse_design_result_path)
    print('finished')


if __name__ == "__main__":
    main()
