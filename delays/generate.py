# Generate the network matrix for sensor-gateway pairs
# delay_client_to_gateway, a nxm matrix. The (i,j) element is the round delay
# between client i and gateway j, including downlink, comp time and uplink
# delays. The (i,j) element is zero if the connection between i and j is not
# feasible.
#
# delay_gateway_to_cloud, a mx1 vector recording the comm delay from gateway to
# cloud
#
# delay_device, a nx1 vector recording the computational delay on each device

import numpy as np
import random

gateway_num = 6
sensor_num = 184

"""
model_size = 1600  # kB
delay_min = model_size / 5000
delay_max = model_size / 200

delay_client_to_gateway = None
#gateway_ids = ((gateway_num-1)/2 + 1 * np.random.standard_normal(sensor_num)).astype(int)
#gateway_ids = np.clip(gateway_ids, 0, gateway_num-1)
gateway_ids = np.random.randint(gateway_num, size=(sensor_num,))
print(gateway_ids)
for i in range(sensor_num):
    delay_i = random.uniform(delay_min, delay_max)

    if delay_client_to_gateway is None:
        delay_client_to_gateway = \
            np.eye(gateway_num)[gateway_ids[i]].reshape((1, -1)) * delay_i
    else:
        delay_client_to_gateway = np.concatenate(
            (delay_client_to_gateway,
             np.eye(gateway_num)[gateway_ids[i]].reshape((1, -1)) * delay_i),
            axis=0
        )
print(delay_client_to_gateway)
np.savetxt('comm_sr_to_gw.txt', delay_client_to_gateway)

delay_to_cloud_min = 18.72
delay_to_cloud_max = 18.72
delay_gateway_to_cloud = np.random.uniform(low=delay_to_cloud_min,
                                           high=delay_to_cloud_max,
                                           size=(gateway_num,))
np.savetxt('comm_gw_to_cl.txt', delay_gateway_to_cloud)
"""

comp_delay_mean = 4
comp_delay_sigma = 3
delay_device = np.random.lognormal(comp_delay_mean, comp_delay_sigma, size=(sensor_num,)) + 30
delay_device = np.minimum(delay_device, 15000)
print(delay_device)
np.savetxt('comp1.txt', delay_device)
