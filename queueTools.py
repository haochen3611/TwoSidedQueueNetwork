import queueing_tool as qt
import numpy as np
import networkx as nx

adj_list = {0: [1], 1: [0, 3], 2: [0, 3], 3: [2]}
edg_list = {0: {1: 1}, 1: {0: 2, 3: 2}, 2: {0: 2, 3: 2}, 3: {2: 1}}

q_classes = {1: qt.ResourceQueue,
             2: qt.QueueServer}

routing_matrix = np.array([[0, 1], [1, 0]])
horizon = 10000
arrival_rates = np.array([0.1, 0.1])  # expected passenger arrival rate count/sec
trip_mean_time = np.array([[1, 1000], [500, 1]])  # expected trip time
initial_vehicle = np.array([20, 20])

q_args = {
    1: {
        "num_servers": 200,
        "qbuffer": 2000,
        "arrival_f": lambda t: t + np.random.exponential(10),
        "service_f": lambda t: t,
        "AgentFactory": qt.ResourceAgent
    },
    2: {
        "num_servers": np.infty,
        "service_f": lambda t: t + np.random.exponential(500)
    }
          }

graph = qt.adjacency2graph(adjacency=adj_list, edge_type=edg_list)
net = qt.QueueNetwork(g=graph,
                      q_classes=q_classes,
                      q_args=q_args,
                      seed=13)
# r_mat = qt.generate_transition_matrix(graph, seed=100)
net.set_transitions({1: {3: 1}, 2: {0: 1}})
print(net.transitions(False))

net.initialize(edge_type=1)
print(net.num_events)
# net.animate(t=horizon,
#             figsize=(5, 5),
#             filename='plots/qt_animation/test.mp4',
#             frames=1000,
#             fps=30,
#             writer='ffmpeg',
#             vertex_size=30)
net.start_collecting_data()
net.simulate(n=400)
net.stop_collecting_data()
print(net.num_events)

data = net.get_queue_data(queues=0)

arrival = data[:, 0]
departure = data[:, 2]

import matplotlib.pyplot as plt

plt.plot(arrival, np.arange(len(arrival)))
plt.plot(departure, np.arange(len(departure)))
plt.legend(['arr', 'dep'])
plt.show()


