grad_val = sess.run(attack.grad,adv_dict_test)
orth_grad_val = np.zeros_like(grad_val)
for i in range(grad_val.shape[0]):
    k = grad_val[i].flatten()
    x = np.random.randn(784)
    x -= x.dot(k) * k / np.linalg.norm(k)**2
    orth_grad_val[i] = x.reshape(1,28,28,1) / np.max(np.abs(x))
    grad_val[i] = grad_val[i]/np.max(np.abs(grad_val[i]))

bound = 0.3
res = 40
xent_img = np.zeros((batch_size, 9, res,res)) # bs, crop#, res_i, res_j 
for i in range(res):
    for j in range(res):
        x_adv = adv_dict_test[input_images] + bound*i/10*grad_val + bound*j/10*orth_grad_val
        xent_i = sess.run(model.xent_indv, {input_images:x_adv, input_label:adv_dict_test[input_label]})
        xent_img[:, :, i, j] = np.asarray(xent_i).transpose((1,0))



from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
fig = plt.figure()
X = [[bound*i/res for i in range(res)] for j in range(res)]
Y = [[bound*j/res for i in range(res)] for j in range(res)]
idx = 1

ax = fig.add_subplot(331, projection='3d')
Z = xent_img[idx,0,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

ax = fig.add_subplot(332, projection='3d')
Z = xent_img[idx,1,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

ax = fig.add_subplot(333, projection='3d')
Z = xent_img[idx,2,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

ax = fig.add_subplot(334, projection='3d')
Z = xent_img[idx,3,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

ax = fig.add_subplot(335, projection='3d')
Z = xent_img[idx,4,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

ax = fig.add_subplot(336, projection='3d')
Z = xent_img[idx,5,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

ax = fig.add_subplot(337, projection='3d')
Z = xent_img[idx,6,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

ax = fig.add_subplot(338, projection='3d')
Z = xent_img[idx,7,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

ax = fig.add_subplot(339, projection='3d')
Z = xent_img[idx,8,:,:]
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Z = np.mean(xent_img[idx],0)
ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap='seismic')
ax.view_init(30, -120)
ax.set_xlabel('$\epsilon_1$')
ax.set_ylabel('$\epsilon_2$')
plt.show()
