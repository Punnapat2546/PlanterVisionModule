import numpy as np
import matplotlib.pyplot as plt


def plot_ellipsoid(means_list, covs_list, name_list):
    fig = plt.figure(figsize=(5 * len(means_list), 4))
    for i in range(len(means_list)):
        means, covs = means_list[i], covs_list[i]
        eigen_values, eigen_vectors = np.linalg.eig(covs)
        ax = fig.add_subplot(1, len(means_list), 1 + i, projection="3d")
        for j in range(len(means)):
            eig_val = eigen_values[j]
            eig_vec = eigen_vectors[j]
            mean = means[j]
            a, b, c = np.sqrt(eig_val)
            theta = np.linspace(0, np.pi, 50)
            phi = np.linspace(0, 2 * np.pi, 100)
            Theta, Phi = np.meshgrid(theta, phi)
            x1 = a * np.sin(Theta) * np.cos(Phi)
            y1 = b * np.sin(Theta) * np.sin(Phi)
            z1 = c * np.cos(Theta)
            points = np.stack([t.flatten() for t in [x1, y1, z1]])
            v1 = eig_vec[:, 0]
            v2 = eig_vec[:, 1]
            v3 = eig_vec[:, 2]
            T = np.array([v1, v2, v3]).T
            new_points = T @ points
            x2 = new_points[0, :]
            y2 = new_points[1, :]
            z2 = new_points[2, :]
            x2, y2, z2 = [t.reshape(x1.shape) for t in [x2, y2, z2]]
            surf = ax.plot_wireframe(
                z2 + mean[2],
                y2 + mean[1],
                x2 + mean[0],
                rstride=5,
                cstride=2,
                color="brown",
                linewidth=0.2,
            )
            ax.scatter3D(mean[2], mean[1], mean[0], color="black", marker="o")
        ax.set_title(name_list[i])
        ax.axes.set_xlim3d(left=-0.5, right=0.5)
        ax.axes.set_ylim3d(bottom=-0.5, top=0.5)
        ax.axes.set_zlim3d(bottom=0, top=1)
        ax.set_xlabel("Cr"), ax.set_ylabel("Cb"), ax.set_zlabel("Y")
        ax.view_init(elev=7, azim=-7)


def plot_ellipsoid_color_model(means_list, covs_list, name_list):
    fig = plt.figure(figsize=(5 * 20, 4 * (len(means_list) // 20 + 1)))
    for i in range(len(means_list)):
        means, covs = means_list[i], covs_list[i]
        eigen_values, eigen_vectors = np.linalg.eig(covs)
        ax = fig.add_subplot((len(means_list) // 20 + 1), 20, 1 + i, projection="3d")
        for j in range(len(means)):
            eig_val = eigen_values[j]
            eig_vec = eigen_vectors[j]
            mean = means[j]
            a, b, c = np.sqrt(eig_val)
            theta = np.linspace(0, np.pi, 50)
            phi = np.linspace(0, 2 * np.pi, 100)
            Theta, Phi = np.meshgrid(theta, phi)
            x1 = a * np.sin(Theta) * np.cos(Phi)
            y1 = b * np.sin(Theta) * np.sin(Phi)
            z1 = c * np.cos(Theta)
            points = np.stack([t.flatten() for t in [x1, y1, z1]])
            v1 = eig_vec[:, 0]
            v2 = eig_vec[:, 1]
            v3 = eig_vec[:, 2]
            T = np.array([v1, v2, v3]).T
            new_points = T @ points
            x2 = new_points[0, :]
            y2 = new_points[1, :]
            z2 = new_points[2, :]
            x2, y2, z2 = [t.reshape(x1.shape) for t in [x2, y2, z2]]
            co = "green" if j == 0 else "brown"
            surf = ax.plot_wireframe(
                z2 + mean[2],
                y2 + mean[1],
                x2 + mean[0],
                rstride=5,
                cstride=2,
                color=co,
                # color="brown",
                linewidth=0.2,
            )
            ax.scatter3D(mean[2], mean[1], mean[0], color="black", marker="o")
        ax.set_title(name_list[i])
        ax.axes.set_xlim3d(left=-0.5, right=0.5)
        ax.axes.set_ylim3d(bottom=-0.5, top=0.5)
        ax.axes.set_zlim3d(bottom=0, top=1)
        ax.set_xlabel("Cr"), ax.set_ylabel("Cb"), ax.set_zlabel("Y")
        ax.view_init(elev=7, azim=-7)
