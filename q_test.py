import queueing_tool as qt
import numpy as np
import matplotlib.pyplot as plt

adj_list = {0: [1], 1: [0]}
edg_list = {0: {1: 1}, 1: {0: 2}}
q_cls = {1: qt.ResourceQueue,
         2: qt.QueueServer}
q_args = {
    1: {
        "num_servers": 1000,
        "arrival_f": lambda t: t + np.random.exponential(500),
        "service_f": lambda t: t,
        "AgentFactory": qt.ResourceAgent,
        "qbuffer": 1000,
        "active_cap": np.infty
    },
    2: {
            "num_servers": np.infty,
            "service_f": lambda t: t + np.random.exponential(1)
        }
}

net = qt.QueueNetwork(g=qt.adjacency2graph(adjacency=adj_list, edge_type=edg_list),
                      q_classes=q_cls,
                      q_args=q_args,
                      seed=66)
net.initialize(edges=(0, 1))
print(net.num_events)
# net.animate(t=horizon,
#             figsize=(5, 5),
#             filename='plots/qt_animation/test.mp4',
#             frames=1000,
#             fps=30,
#             writer='ffmpeg',
#             vertex_size=30)
net.start_collecting_data()
for _ in range(1000):
    net._simulate_next_event(slow=False)
    q1 = net.edge2queue[0]
    q2 = net.edge2queue[1]
    print('num_servers:', q1.num_servers)
    print('Q1 arrivals:', q1.num_arrivals)
    print('Q2 arrivals', q2.num_arrivals)
    print('Q2 departures: ', q2.num_departures)
net.stop_collecting_data()
print(net.num_events)

data = net.get_queue_data(queues=0)

arrival = np.sort(data[:, 0])
departure = np.sort(data[:, 2])

plt.plot(arrival, np.arange(len(arrival)))
plt.plot(departure, np.arange(len(departure)))
plt.legend(['arr', 'dep'])
plt.show()
