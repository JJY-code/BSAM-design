import numpy as np
import os

dir = r'C:\AFM_revision\batch_12'
num_samples = 900


def get_data_uniaxial_z():
    data_all_RF3 = []
    data_all_U1 = []
    for i in range(num_samples):
        path_sample = os.path.join(dir, str(i))
        path_data_RF3 = os.path.join(path_sample, 'data_RF3_uniaxial_z.txt')
        path_data_U1 = os.path.join(path_sample, 'data_U1_uniaxial_z.txt')
        data_RF3 = np.loadtxt(path_data_RF3)
        data_U1 = np.loadtxt(path_data_U1)
        data_all_RF3 = data_all_RF3 + [data_RF3[:, 1]]
        data_all_U1 = data_all_U1 + [data_U1[:, 1]]
    data_all_RF3 = np.array(data_all_RF3)
    data_all_U1 = np.array(data_all_U1)
    np.savetxt('data_all_RF3_uniaxial_z.txt', data_all_RF3)
    np.savetxt('data_all_U1_uniaxial_z.txt', data_all_U1)


def get_data_uniaxial_y():
    data_all_RF2 = []
    data_all_U1 = []
    data_all_U3 = []
    for i in range(num_samples):
        path_sample = os.path.join(dir, str(i))
        path_data_RF2 = os.path.join(path_sample, 'data_RF2_uniaxial_y.txt')
        path_data_U1 = os.path.join(path_sample, 'data_U1_uniaxial_y.txt')
        path_data_U3 = os.path.join(path_sample, 'data_U3_uniaxial_y.txt')
        data_RF2 = np.loadtxt(path_data_RF2)
        data_U1 = np.loadtxt(path_data_U1)
        data_U3 = np.loadtxt(path_data_U3)
        data_all_RF2 = data_all_RF2 + [data_RF2[:, 1]]
        data_all_U1 = data_all_U1 + [data_U1[:, 1]]
        data_all_U3 = data_all_U3 + [data_U3[:, 1]]
    data_all_RF2 = np.array(data_all_RF2)
    data_all_U1 = np.array(data_all_U1)
    data_all_U3 = np.array(data_all_U3)
    np.savetxt('data_all_RF2_uniaxial_y.txt', data_all_RF2)
    np.savetxt('data_all_U1_uniaxial_y.txt', data_all_U1)
    np.savetxt('data_all_U3_uniaxial_y.txt', data_all_U3)


def get_data_biaxial():
    data_all_RF2_biaxial = []
    data_all_RF3_biaxial = []
    data_all_U1_biaxial = []
    for i in range(num_samples):
        path_sample = os.path.join(dir, str(i))
        path_data_rf2_biaxial = os.path.join(path_sample, 'data_RF2_biaxial.txt')
        path_data_rf3_biaxial = os.path.join(path_sample, 'data_RF3_biaxial.txt')
        path_data_u1_biaxial = os.path.join(path_sample, 'data_U1_biaxial.txt')
        data_rf2_biaxial = np.loadtxt(path_data_rf2_biaxial)
        data_rf3_biaxial = np.loadtxt(path_data_rf3_biaxial)
        data_u1_biaxial = np.loadtxt(path_data_u1_biaxial)
        data_all_RF2_biaxial = data_all_RF2_biaxial + [data_rf2_biaxial[:, 1]]
        data_all_RF3_biaxial = data_all_RF3_biaxial + [data_rf3_biaxial[:, 1]]
        data_all_U1_biaxial = data_all_U1_biaxial + [data_u1_biaxial[:, 1]]
    data_all_RF2_biaxial = np.array(data_all_RF2_biaxial)
    data_all_RF3_biaxial = np.array(data_all_RF3_biaxial)
    data_all_U1_biaxial = np.array(data_all_U1_biaxial)
    np.savetxt('data_all_RF2_biaxial.txt', data_all_RF2_biaxial)
    np.savetxt('data_all_RF3_biaxial.txt', data_all_RF3_biaxial)
    np.savetxt('data_all_U1_biaxial.txt', data_all_U1_biaxial)


# def get_data_biaxial():


if __name__ == '__main__':
    get_data_biaxial()
