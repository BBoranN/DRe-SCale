import math
from datetime import datetime
import time as time
import json as json
import setting
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

# Utility to connect to K8s API, Prometheus API and OpenFaaS API
config.load_config()
deployment_name = 'matmul' # deployment name for deployed function
namespace = 'openfaas-fn' # default namespace for openfaas functions
scale_api = client.AppsV1Api()
resource_usage_api = client.CustomObjectsApi()
prom = PrometheusConnect(url=setting.ppo_lstm_agent_prometheus_url, disable_ssl=True)


class Environment(gym.Env):

    # every environment should support None render mode
    metadata = {'render_modes': ['human', None]}

    def __init__(self, rew_range=(-100, 10000), min_pods=1, max_pods=24) -> None:
        super(Environment, self).__init__()

        
        # FIXED PARAMETERS / Configurable
        self.reward_range = rew_range
        self.MAX_PODS = max_pods
        self.MIN_PODS = min_pods

        self.sampling_window = 30 
        self.timestep = 0
        self.episode = 0
        self.loop = 0
        self._last_obs = None
        self._stats_window = 100

        self.reward_history = []
        self.score = 0
        # [avg_execution, throughput, requests, replicas, avg_cpu/req, avg_mem/req]
        self.observation_space = spaces.Box(
                                    low=np.array([0, 0, 0, self.MIN_PODS, 0, 0]), 
                                    high=np.array([60, 100, 100, self.MAX_PODS, 2, 2]), 
                                    shape=(6,), 
                                    dtype=np.float64)
        
        # action is either increase or decrease pods
        self.action_space = spaces.Discrete(5)
        self._action_to_scale = {0: -2,
                                 1: -1,
                                 2: 0,
                                 3: 1,
                                 4: 2}
        self._initial_setup()

        
 

    def _initial_setup(self):
        # function resource requests
        self.func_cpu = 150 # millicores
        self.func_mem = round((256/1024), 2) # in GBi

        # custom metrics from env
        logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(logdir + "/${PPO_LSTM_Run3_cpu}")
        self.file_writer.set_as_default() 

        self._reward_file = 'reward_history_PPO_LSTM_Run3_cpu.json'

    def _get_info(self):
        return {}
    
    # Utility function to take action in the environment
    def _take_action(self, action):
        try:
            # number of ready function replicas 
            current_pods = scale_api.read_namespaced_deployment(name=deployment_name,
                                            namespace=namespace).status.ready_replicas
            if current_pods == None:
                current_pods = 0
                print('current pods are NoneType')
        except Exception as e:
            current_pods = 0
            print('Error in reading ready pods')

        scale_value = current_pods + action
        action_feedback = False

        if action < 0 :
            if (scale_value >= self.MIN_PODS):
                action_feedback = True
                body = {'spec': {'replicas': scale_value}}
                try:
                    _ = scale_api.patch_namespaced_deployment_scale(name=deployment_name, 
                                                    namespace=namespace,
                                                    body=body).spec.replicas
                except Exception as e:
                    action_feedback = False
                    print(e)
            else:
                action_feedback = False

        elif action == 0:
            if scale_value == 0:
                action_feedback = True
            else:
                action_feedback = False

        else:
            if (scale_value <= self.MAX_PODS and scale_value >= self.MIN_PODS):
                action_feedback = True
                body = {'spec': {'replicas': scale_value}}
                try:
                    _ = scale_api.patch_namespaced_deployment_scale(name=deployment_name, 
                                                        namespace=namespace,
                                                        body=body).spec.replicas
                except Exception as e:
                    action_feedback = False
                    print(e)
            else:
                action_feedback = False
             
        info = {'action': action, 
               'action_feedback': action_feedback, 
               'pods': current_pods,
               'scale_value': scale_value}
        
        return info

    def _get_obs(self):
        # Initialize default values to avoid UnboundLocalError
        avg_execution = 0.0
        replicas = 0
        requests = 0
        throughput = 100
        avg_cpu = 0.0
        avg_mem = 0.0

        # 1. Get the avg execution time
        query1 = f"(rate(gateway_functions_seconds_sum{{function_name='{deployment_name}.{namespace}', code='200'}}[30s]) / rate(gateway_functions_seconds_count{{function_name='{deployment_name}.{namespace}', code='200'}}[30s]))"
        try:
            data = prom.custom_query(query=query1)
            if data and 'value' in data[0] and len(data[0]['value']) > 1:
                val = float(data[0]['value'][1])
                avg_execution = 0.0 if math.isnan(val) else round(val, 3)
            else:
                print("PromQL query1 (avg_execution) returned empty.")
        except Exception as e:
            print(f"Error in avg_execution query: {e}")

        # 2. Get ready replicas via K8s API directly (more reliable than Prometheus)
        try:
            deploy = scale_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            if deploy.status.ready_replicas is not None:
                replicas = deploy.status.ready_replicas
            else:
                replicas = 0
                print("Replicas returned None (likely 0 ready pods)")
        except Exception as e:
            replicas = 0
            print(f"Error reading replicas from K8s API: {e}")

        # 3. Get total requests
        query4 = f"increase(gateway_function_invocation_total{{function_name='{deployment_name}.{namespace}'}}[30s])"
        try:
            data = prom.custom_query(query=query4)
            if data:
                for d in data:
                    if 'value' in d and len(d['value']) > 1:
                        requests += int(float(d['value'][1]))
            else:
                print("PromQL query4 (requests) returned empty.")
        except Exception as e:
            print(f"Error in requests query: {e}")
            
        print(f'Requests during window: {requests}')

        # 4. Get throughput (percentage of successful requests)
        query2 = f"increase(gateway_function_invocation_total{{code='200', function_name='{deployment_name}.{namespace}'}}[30s])"
        try:
            data = prom.custom_query(query=query2)
            successful_requests = 0
            if data and 'value' in data[0] and len(data[0]['value']) > 1:
                successful_requests = int(float(data[0]['value'][1]))
            
            if requests > 0:
                throughput = int(round((successful_requests / requests) * 100, 2))
            else:
                throughput = 100 # Default to 100% if no requests were made
        except Exception as e:
            print(f"Error in throughput query: {e}")
            throughput = 100 if requests == 0 else 0

        # 5. Get Pod Resource Metrics (CPU/Memory)
        try:
            resource_list = resource_usage_api.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", namespace, "pods")
            
            # Safely parse the pods dict
            my_pods = [
                pod['containers'][0]['usage'] 
                for pod in resource_list.get('items', []) 
                if pod.get('metadata', {}).get('labels', {}).get('faas_function') == deployment_name
            ]
            
            cpu_total = 0
            mem_total = 0
            
            if my_pods:
                for pod_usage in my_pods:
                    c = pod_usage.get('cpu', '0')
                    m = pod_usage.get('memory', '0')
                    
                    # Parse CPU
                    try:
                        if c.endswith('n'): cpu_total += (round(int(c.split('n')[0])/1e6, 4))
                        elif c.endswith('u'): cpu_total += (round(int(c.split('u')[0])/1e3, 4))
                        elif c.endswith('m'): cpu_total += (round(int(c.split('m')[0]), 4))
                    except Exception:
                        pass
                        
                    # Parse Memory
                    try:
                        if m.endswith('Ki'): mem_total += (round(int(m.split('Ki')[0])/(1024*1024), 4))
                        elif m.endswith('Mi'): mem_total += (round(int(m.split('Mi')[0])/1024, 4))
                        elif m.endswith('Gi'): mem_total += (round(int(m.split('Gi')[0]), 4))
                    except Exception:
                        pass
                
                avg_cpu = round((cpu_total / len(my_pods)) / self.func_cpu, 4)
                avg_mem = round((mem_total / len(my_pods)) / self.func_mem, 4)
            else:
                print("No pods found matching the deployment name for metrics.")

        except Exception as e:
            print(f'Error fetching pod metrics from k8s API: {e}')
            
        # Compile and return observation
        obs = np.array([avg_execution, throughput, requests, replicas, avg_cpu, avg_mem])
        return obs
        
    def _write_to_board(self, obs, action, rew, info, step, episode):
        # write to tensorboard
        with self.file_writer.as_default():
            tf.summary.scalar('avg_execution_time', obs[0], step)
            tf.summary.scalar('throughput', obs[1], step)
            tf.summary.scalar('requests', obs[2], step)
            tf.summary.scalar('replicas', obs[3], step)
            tf.summary.scalar('cpu', obs[4], step)
            tf.summary.scalar('mem', obs[5], step)
            tf.summary.scalar('episode', episode, step)
            tf.summary.scalar('action', (action), step)
            if info['action_feedback']:
                tf.summary.scalar('action_feedback', 1 , step)
            else:
                tf.summary.scalar('action_feedback', 0 , step)
            tf.summary.scalar('n-step_reward', rew, step)


    # calculate and return reward based on the observation
    def _calculate_reward(self, obs, metadata={}):
        reward = 0
        meta_scale_value = metadata['scale_value']
        throughput = obs[1] # %
        _ = obs[2]
        replicas = obs[3]
        avg_cpu = obs[4] # % 0 - 1
        avg_mem = obs[5] # % 0 - 1


        alpha = 0.75
        beta = 0.125
        gamma = 0.125
        phi = 0.25
        r_th = alpha * (throughput ** 2)
        r_cpu = beta * (avg_cpu*100)
        r_mem = gamma * (avg_mem*100)
        r_rep = -phi * ((replicas - self.MIN_PODS) ** 2)

        reward = r_th + r_cpu + r_mem + r_rep
        reward = round(reward, 2)

        # action unsuccessful
        if (meta_scale_value != replicas):
            reward += self.reward_range[0]
        
        return reward
    


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # reset other paramters based on the environment
        self.score = 0
        self.loop = 0
        observation = self._get_obs()
        info = self._get_info()
        self._last_obs = observation
      
        return observation, info


    def step(self, action):
        done = False
        # Map the action (element of {0,1,2,3,4}) to scaling
        action = self._action_to_scale[action]

        # execute the action in environment
        info = self._take_action(action=action)

        # immediate negative reward - invalid action
        if info['action_feedback'] == False:
            self._write_to_board(self._last_obs, action, -100, info, self.timestep, self.episode)
            self.timestep += 1
            self.loop += 1
            self.score += -100
            if self.loop == 10:
                done = True
            return self._last_obs, -100, done, False, info 
        else:
            # wait for the sampling window to get the next observation
            time.sleep(self.sampling_window)

            # get the next observation
            next_obs = self._get_obs()
            # calculate reward
            reward = self._calculate_reward(obs=next_obs, metadata=info)
            self.score += round(reward, 2)
            self._write_to_board(next_obs, action, reward, info, self.timestep, self.episode)

            # counter for custom metrics
            self.timestep += 1
            if (self.timestep % 10 == 0):
                done = True
                self.episode += 1
                self.loop = 0
                self.reward_history.append(self.score)
                with self.file_writer.as_default():
                    tf.summary.scalar('episodic_reward', self.score, self.episode)
                    tf.summary.scalar('mean_reward', np.mean(self.reward_history[-self._stats_window:]), self.episode)
                self.score = 0
                history = {'reward_history': self.reward_history,
                              'last_episode': self.episode}
                # write the reward history to a file
                with open(self._reward_file, "w") as outfile:
                    json.dump(history, outfile)
                
            self._last_obs = next_obs

            return next_obs, reward, done, False, info
    
    
    def render(self, mode='human', close=False):
        # render or print information on screen or add to the tensorboard, etc.
        pass

    def close(self):
        # close any open resources
        pass