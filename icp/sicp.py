
#%%
# pip install simpleicp 
from simpleicp import PointCloud, SimpleICP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Read point clouds from xyz files into n-by-3 numpy arrays
X_fix = np.genfromtxt("bunny_part1.xyz")
X_mov = np.genfromtxt("bunny_part2.xyz")

#%% 
Data = np.load('icp_data.npz')
Line1 = Data['LineGroundTruth']
Line2 = Data['LineMovedCorresp']
Line1 = np.vstack([Line1, np.zeros((1,Line1.shape[1]))])
Line2 = np.vstack([Line2, np.zeros((1,Line2.shape[1]))])
Line1 = Line1.T
Line2 = Line2.T
dfLine1 = pd.DataFrame(Line1, columns=["x", "y", "z"])
dfLine2 = pd.DataFrame(Line2, columns=["x", "y", "z"])
print(dfLine1)

pcLine1 = PointCloud(dfLine1)
pcLine2 = PointCloud(dfLine2)
print(pcLine1)


# Create point cloud objects
pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

print(pc_fix[0:5])


#%% 
# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
# icp.add_point_clouds(pc_fix, pc_mov)
icp.add_point_clouds(pcLine1, pcLine2) 
H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = \
        icp.run(max_overlap_distance=1)


#%% 
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
# ax.set_xlim(axes_lims[0], axes_lims[3])
# ax.set_ylim(axes_lims[1], axes_lims[4])
# ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# ax.scatter(*Line1.T, color='red')
# ax.scatter(*Line2.T, color='blue')

fix_sel = np.array(pc_fix[pc_fix["selected"]==True][["x", "y", "z"]])

#ax.scatter(*X_fix.T, s=1, color='red')
ax.scatter(*X_mov.T, s=1, color='blue')
ax.scatter(*fix_sel.T, s=5, color='green')


plt.show()

