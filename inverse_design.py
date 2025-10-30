import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from Design_AFM_revision.ForwardPrediction.nn_uniaxial_z_rf.prediction import MyModel, DataLoader, \
    predict, COVID19Dataset
import torch
import math
from sko.GA import RCGA
from sko.PSO import PSO
from functools import partial
from scipy.interpolate import interp1d
import scipy.io

num_features = 8
num_output_dims = 21
device = 'cuda'
path_all_NN = r'D:\ResearchDoc\MachineLearing\PycharmDoc\CB_inversedesign\Design_0915\NN_1004'


def rf2stress(rf):
    # should not to divide 0.3, nominal stress: \sigma = F/A
    # 之前除以0.3进行的优化
    stress = 4 * 1 * 1000 / 19.5 / 19.5 * rf / 0.3
    return stress


def deep_learn_predict(input, model):
    """
    :param input: the structure parameters, num_samples * features (ndarray)
    :param model: the ANN model used to predict the mechanical response
    :return: the predict mechanical response
    """
    # fix batch size to  all data_x
    num_samples = input.shape[0]
    dataset_input = COVID19Dataset(input)
    input_loader = DataLoader(dataset_input, batch_size=num_samples, shuffle=False, pin_memory=True)
    preds = predict(input_loader, model, device)
    return rf2stress(preds)


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
    # 预测结果在目标点上插值


def design_target_displace():
    array_x = np.linspace(1 / 20, 1, 20)
    # x*cos
    # array_y = np.cos(array_x * np.pi) * array_x * 5
    # sin
    array_y = np.sin(array_x * np.pi) * 3
    # linear
    # array_y = array_x * 5
    # tanh
    # array_y = np.tanh(array_x*np.pi)*5
    # cos
    # array_y = (1-np.cos(array_x*np.pi))/2*4
    # array_y = my_target_fun(array_x)
    target = np.vstack((array_x, array_y))
    # target = np.array([[0.2, 0.5, 1], [2, -2, 2]])
    return target


def design_target_displace_adjustable(A, Omega):
    # strain range
    strain_start = 0
    strain_end = 3
    strain_range = np.linspace(strain_start, strain_end, 21).reshape(1, -1)

    x_0to1 = (strain_range - strain_start) / (strain_end - strain_start)
    # sin
    target_displacement = A * np.sin(x_0to1 * Omega)
    data_target = np.vstack((strain_range, target_displacement))
    return data_target


def target():
    # strain range
    strain_start = 0
    strain_end = 3
    strain_range = np.linspace(strain_start, strain_end, 21).reshape(1, -1)
    # # transform strain_start to strain_end to [-1, 1]
    # x = (strain_range-strain_start)/(strain_end - strain_start)*2-1
    x_0to1 = (strain_range - strain_start) / (strain_end - strain_start)
    # mixed
    # target_displacement = 1 * np.sin(x_0to1 * np.pi * 3)
    # target_displacement = 3 * (1 - np.cos(2.5 * np.pi * x_0to1)) + 4 * np.sin(np.pi / 2 * x_0to1)
    # # sin
    target_displacement = 5 * np.sin(x_0to1 * np.pi * 1)
    # # asin
    # # target_displacement = 5 * (np.arcsin(x)+np.pi/2)/np.pi
    # # # square
    # # target_displacement = 10*(np.sign(strain_range-1)+1)/2
    # # pulse
    # # target_displacement = 10 * (1-np.cos(np.maximum(0, strain_range-1.6)*2.5*np.pi/2))
    # # complex
    # # target_displacement = 5 * strain_range*(strain_range-1.2)*(strain_range-2.5)
    # # double freq
    # # target_displacement = 20*(1 - np.cos((np.arcsin(x) / np.pi * 2 + 1) * np.pi * 1.)) / 2
    # # # trinary freq
    # # target_displacement = 5*(1 - np.cos((np.arcsin(x) / np.pi * 2 + 1) * np.pi * 1.5)) / 2
    # # # flip
    # # target_displacement = 10*((strain_range-(strain_end+strain_start)/2)*np.sign(strain_range-(strain_end+strain_start)/2)-1)
    # # filter
    # # target_displacement = 6*np.maximum(x, 0)
    # # # fourfold freq
    # # target_displacement = 1*(1 - np.cos((np.arcsin(x) / np.pi * 2 + 1) * np.pi * 2)) / 2
    data_target = np.vstack((strain_range, target_displacement))
    # # data_target = np.append(data_target, np.array([[2.1], [12]]), axis=1)
    # # 离散点
    # data_target = np.array([[0.5, 1.5, 2.25, 3.], [1., -1.5, 2., -3]])

    return data_target


def my_target_fun(x):
    y = x.copy()
    y[x <= 0.5] = x[x <= 0.5] * 8
    y[x > 0.5] = 4
    return y


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
    pso = PSO(func=pso_design_displace_target_function_fix_para, n_dim=num_feature_solve, pop=128, max_iter=18,
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
    # np.savetxt(os.path.join(inverse_design_result_path, 'pred_1.txt'), pred_1)
    # np.savetxt(os.path.join(inverse_design_result_path, 'target_1.txt'), target_1, )
    #
    # pred_2 = deep_learn_predict(best_x.reshape(1, -1), model_2)
    # # np.savetxt(os.path.join(inverse_design_result_path, 'pred_1.txt'), pred_2)
    # # np.savetxt(os.path.join(inverse_design_result_path, 'target_1.txt'), target_2, )

    # figure_plot_the best response and target
    # ax_res.plot(np.linspace(0.0, 0.3, 21), pred_1[0])
    # ax_res.plot(np.linspace(0.0, 0.3, 21), pred_2[0])
    # ax_res.plot(target_1[:, 0], target_1[:, 1], '*')
    # ax_res.plot(target_2[:, 0], target_2[:, 1], '*')
    # plt.legend(('pred_1', 'target_1', 'pred_2', 'target_2'))
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
    rf_z_muscle_original = np.loadtxt('./data/skeletal muscle z.txt')
    rf_y_muscle_original = np.loadtxt('./data/skeletal muscle y.txt')
    rf_z_muscle_biaxial_original = np.loadtxt('./data/skeletal muscle biaxial z.txt')
    rf_y_muscle_biaxial_original = np.loadtxt('./data/skeletal muscle biaxial y.txt')
    rf_z_brain = np.loadtxt('./data/brain z.txt')
    rf_y_brain = np.loadtxt('./data/brain y.txt')
    rf_z_kidney = np.loadtxt('./data/kidney z.txt')
    rf_y_kidney = np.loadtxt('./data/kidney y.txt')
    stress_vessels_l = np.loadtxt('./data/data_vessel_L.txt')
    stress_vessels_t = np.loadtxt('./data/data_vessel_T.txt')
    stress_pericardium_y = np.loadtxt('./data/data_pericardium_y.txt')
    stress_pericardium_z = np.loadtxt('./data/data_pericardium_z.txt')

    # remover the rows which the excess the design range
    rf_z_muscle = rf_z_muscle_original[(rf_z_muscle_original[:, 0] < 0.3) & (rf_z_muscle_original[:, 0] > 0.0), :]
    rf_y_muscle = rf_y_muscle_original[(rf_y_muscle_original[:, 0] < 0.3) & (rf_y_muscle_original[:, 0] > 0.0), :]
    rf_z_muscle_biaxial = rf_z_muscle_biaxial_original[(rf_z_muscle_biaxial_original[:, 0] < 0.3) & (rf_z_muscle_biaxial_original[:, 0] > 0.0), :]
    rf_y_muscle_biaxial = rf_y_muscle_biaxial_original[(rf_y_muscle_biaxial_original[:, 0] < 0.3) & (rf_y_muscle_biaxial_original[:, 0] > 0.0), :]

    np.random.seed(112)
    # target_displace = design_target_displace_adjustable(3, np.pi*2)
    target_curve_z = rf_z_muscle
    target_curve_y = rf_y_muscle
    model_para_path_z = (r'D:\ResearchDoc\MachineLearing\PycharmDoc\CB_inversedesign\Design_AFM_revision'
                         r'\ForwardPrediction\nn_uniaxial_z_rf\models\model_rf_z.ckpt')
    model_para_path_y = (r'D:\ResearchDoc\MachineLearing\PycharmDoc\CB_inversedesign\Design_AFM_revision'
                         r'\ForwardPrediction\nn_uniaxial_y_rf\models\model_rf_z.ckpt')
    model_para_path_biaxial_z = (r'D:\ResearchDoc\MachineLearing\PycharmDoc\CB_inversedesign\Design_AFM_revision'
                                 r'\ForwardPrediction\nn_biaxial_z_rf\models\model_rf_z.ckpt')
    model_para_path_biaxial_y = (r'D:\ResearchDoc\MachineLearing\PycharmDoc\CB_inversedesign\Design_AFM_revision'
                                 r'\ForwardPrediction\nn_biaxial_y_rf\models\model_rf_z.ckpt')

    model_z = get_pred_model(model_para_path_z)
    model_y = get_pred_model(model_para_path_y)
    model_biaxial_z = get_pred_model(model_para_path_biaxial_z)
    model_biaxial_y = get_pred_model(model_para_path_biaxial_y)
    model_target_pair_z = ModelTargetPair(model_z, target_curve_z)
    model_target_pair_y = ModelTargetPair(model_y, target_curve_y)
    model_target_pair_biaxial_z = ModelTargetPair(model_biaxial_z, rf_z_muscle_biaxial)
    model_target_pair_biaxial_y = ModelTargetPair(model_biaxial_y, rf_y_muscle_biaxial)
    model_target_pair_brain_z = ModelTargetPair(model_z, rf_z_brain)
    # model_target_pair_brain_y = ModelTargetPair(model_y, rf_y_brain)
    model_target_pairs_list = [model_target_pair_z, model_target_pair_y, model_target_pair_biaxial_z,
                                      model_target_pair_biaxial_y]
    model_target_pair_kidney_z = ModelTargetPair(model_z, rf_z_kidney)
    model_target_pair_kidney_y = ModelTargetPair(model_y, rf_y_kidney)
    model_target_pair_vessel_z = ModelTargetPair(model_z, stress_vessels_l)
    model_target_pair_vessel_y = ModelTargetPair(model_y, stress_vessels_t)
    model_target_pair_pericardium_y = ModelTargetPair(model_y, stress_pericardium_y)
    model_target_pair_pericardium_z = ModelTargetPair(model_z, stress_pericardium_z)
    # model_target_pairs_list_kidney = [model_target_pair_kidney_z, model_target_pair_kidney_y]
    # model_target_pairs_list = [model_target_pair_brain_z, model_target_pair_brain_y]
    # model_target_pairs_list = [model_target_pair_vessel_z, model_target_pair_vessel_y]
    # model_target_pairs_list = [model_target_pair_pericardium_y, model_target_pair_pericardium_z]
    # inverse_design_result_path = './res_muscle_062923'
    inverse_design_result_path = './res_muscle_102819'
    os.makedirs(inverse_design_result_path, exist_ok=True)
    best_sample = main_pso_optimize(model_target_pairs_list, inverse_design_result_path)

    # best_sample = np.loadtxt(os.path.join(inverse_design_result_path, 'feature.txt'))
    # plot_res_best(best_sample, model_target_pairs_list=model_target_pairs_list,
    #               inverse_design_result_path=inverse_design_result_path)
    # print('finished')


if __name__ == "__main__":
    main()
