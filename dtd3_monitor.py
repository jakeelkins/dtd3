'''
Distributed TD3 monitor function. Pulls out the agent status checker from the train loop. Much faster.
'''

import logging
import json
import numpy as np
import time
import os

import grpc

from tensorboardX import SummaryWriter

import dtd3_pb2
import dtd3_pb2_grpc

# ---- setup ----
port = 'localhost:50051'   # !! ENSURE this is same as server

time_rate = 30  # in seconds, interval to ask for a monitor checkup.

logpath = './log/20210227_monitor/'
runname = '20210227_monitor_test1'   # make sure same as client

init_time = time.time()

os.makedirs(logpath, exist_ok=True)

def monitor(monitor_stub):
    global epoch
    
    while True:

        time.sleep(time_rate)

        response = monitor_stub.RunAgentStats(dtd3_pb2.MonitorRequest(status=1))

        curr_rew = response.reward
        curr_ang = response.additional_data

        epoch += 1
        curr_time = time.time()
        writer.add_scalar("curr avg reward", curr_rew, curr_time-init_time)
        writer.add_scalar("curr avg ang", curr_ang, curr_time-init_time)

        print(f'Time elapsed: {round(curr_time-init_time, 2)} s \t Avg reward: {round(curr_rew, 4)} \t Avg ang: {round(curr_ang, 4)}')


if __name__ == '__main__':
    logging.basicConfig()
    print(f'connecting to port {port}...')
    channel = grpc.insecure_channel(port)
    print(f'connected to port {port}. establishing Tensorboard writer...')

    # logging variables:
    writer = SummaryWriter(logpath + runname)
    print(f'Tensorboard is running at path: {logpath + runname}.')

    monitor_stub = dtd3_pb2_grpc.LearnerStub(channel)

    print('now starting monitoring.')

    print('\n ------ [begin] ------ \n')
    epoch = 0

    while True:
        try:
            monitor(monitor_stub)
        except grpc.RpcError as e:
            print(f'[!] ERROR: {e}')
        except KeyboardInterrupt as e:
            print(f'\n ERROR: terminated. {e}')
            print(f'please wait for safe shutdown...')
            # TODO: put extra save here?
            channel.close()
            break