from datetime import datetime
import time as time
import setting
import math
import json as json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

# Utility to connect to K8s API, Prometheus API and OpenFaaS API
config.load_config()
deployment_name = 'matmul' 
namespace = 'openfaas-fn'
scale_api = client.AppsV1Api()
resource_usage_api = client.CustomObjectsApi()
# FIX 1: Removed double 'prom = prom =' typo
prom = PrometheusConnect(url=setting.ppo_lstm_test_agent_prometheus_url, disable_ssl=True)


class Environment(gym.Env):
    metadata = {'render_modes': ['human', None]}

    def __init__(self, rew_range=(-100, 10000), min_pods=1, max_pods=24) -> None:
        super(Environment, self).__init__()
        
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
        self.observation_space = spaces.Box(
                                    low=np.array([0, 0, 0, self.MIN_PODS, 0, 0]), 
                                    high=np.array([60, 100, 100, self.MAX_PODS, 2, 2]), 
                                    shape=(6,), 
                                    dtype=np.float64)
        
        self.action_space = spaces.Discrete(5)
        self._action_to_scale = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        self._initial_setup()

    def _initial_setup(self):
        self.func_cpu = 150 
        self.func_mem = round((256/1024), 2) 

        # FIX 2: Hardcoded model name to avoid ${MODEL_NAME} crash
        model_name = "PPO_LSTM_Run1" 
        logdir = "logs/evaluation/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.file_writer = tf.summary.create_file_writer(logdir + "/" + model_name)
        self.file_writer.set_as_default() 

        self._reward_file = f'evaluation_reward_history_{model_name}.json'

    def _get_info(self):
        return {}
    
    def _take_action(self, action):
        try:
            current_pods = scale_api.read_namespaced_deployment(name=deployment_name,
                                            namespace=namespace).status.ready_replicas
            if current_pods is None:
                current_pods = 0
        except Exception as e:
            current_pods = 0
            print('Error in reading ready pods')

        scale_value = current_pods + action
        action_feedback = False

        # Scaling Logic
        if action < 0 :
            if (scale_value >= self.MIN_PODS):
                action_feedback = True
                body = {'spec': {'replicas': scale_value}}
                try:
                    scale_api.patch_namespaced_deployment_scale(name=deployment_name, 
                                                    namespace=namespace, body=body)
                except Exception as e:
                    action_feedback = False
                    print(e)
            else:
                action_feedback = False
        elif action == 0:
            action_feedback = True
        else:
            if (scale_value <= self.MAX_PODS):
                action_feedback = True
                body = {'spec': {'replicas': scale_value}}
                try:
                    scale_api.patch_namespaced_deployment_scale(name=deployment_name, 
                                                        namespace=namespace, body=body)
                except Exception as e:
                    action_feedback = False
                    print(e)
            else:
                action_feedback = False
             
        info = {'action': action, 'action_feedback': action_feedback, 
               'pods': current_pods, 'scale_value': scale_value}
        return info

    def _get_obs(self):
        # 1. Execution Time
        query1 = "(rate(gateway_functions_seconds_sum{function_name='matmul.openfaas-fn', code='200'}[30s]) / rate(gateway_functions_seconds_count{function_name='matmul.openfaas-fn', code='200'}[30s]))"
        try:
            data = prom.custom_query(query=query1)
            if data:
                avg_execution = round(float((data[0]['value'][1])), 3)
                if math.isnan(avg_execution): avg_execution = 0
            else:
                avg_execution = 0.0
        except Exception:
            avg_execution = 0.0

        # 2. Replicas (Using K8s API)
        try:
            deploy = scale_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            replicas = deploy.status.ready_replicas if deploy.status.ready_replicas else 0
        except Exception as e:
            replicas = 0
            print(f"Error reading replicas: {e}")

        # 3. Total Requests
        try:
            query4 = "increase(gateway_function_invocation_total{function_name='matmul.openfaas-fn'}[30s])"
            data = prom.custom_query(query=query4)
            total = 0
            if data:
                for d in data:
                    total += int(float(d['value'][1]))
            requests = total
        except Exception:
            requests = 0
        print(f'requests are {requests}')

        # 4. Throughput
        try:
            query2 = "increase(gateway_function_invocation_total{code='200', function_name='" + deployment_name + "." + namespace + "'}[30s])"
            data = prom.custom_query(query=query2)
            if data:
                throughput_raw = int(float(data[0]['value'][1]))
                throughput = int(round((throughput_raw/requests)*100, 2)) if requests > 0 else 100
            else:
                throughput = 100 if requests == 0 else 0
        except Exception:
            throughput = 100 if requests == 0 else 0

        # 5. Metrics (CPU/Mem)
        try:
            resource_list = resource_usage_api.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", "openfaas-fn", "pods")
            my_pods  = [pod['containers'][0]['usage'] for pod in resource_list['items'] if pod['metadata']['labels'].get('faas_function') == deployment_name]
            cpu = 0
            mem = 0
            if len(my_pods) > 0:
                for pods in my_pods:
                    c = pods['cpu']
                    m = pods['memory']
                    # CPU conversion
                    if c.endswith('n'): cpu += int(c.replace('n',''))/1e6
                    elif c.endswith('u'): cpu += int(c.replace('u',''))/1e3
                    elif c.endswith('m'): cpu += int(c.replace('m',''))
                    else: cpu += int(c) if c.isdigit() else 0
                    
                    # Memory conversion
                    if m.endswith('Ki'): mem += int(m.replace('Ki',''))/1024/1024
                    elif m.endswith('Mi'): mem += int(m.replace('Mi',''))/1024
                    elif m.endswith('Gi'): mem += int(m.replace('Gi',''))
                    else: mem += int(m) if m.isdigit() else 0

                avg_cpu = round((cpu/len(my_pods))/self.func_cpu, 4)
                avg_mem = round((mem/len(my_pods))/self.func_mem, 4)
            else:
                avg_cpu, avg_mem = 0, 0

        except Exception:
            print('pods not available for metrics')
            avg_cpu, avg_mem = 0, 0
            
        return np.array([avg_execution, throughput, requests, replicas, avg_cpu, avg_mem])
        
    def _write_to_board(self, obs, action, rew, info, step, episode):
        with self.file_writer.as_default():
            tf.summary.scalar('avg_execution_time', obs[0], step)
            tf.summary.scalar('throughput', obs[1], step)
            tf.summary.scalar('requests', obs[2], step)
            tf.summary.scalar('replicas', obs[3], step)
            tf.summary.scalar('cpu', obs[4], step)
            tf.summary.scalar('mem', obs[5], step)
            tf.summary.scalar('episode', episode, step)
            tf.summary.scalar('action', action, step)
            tf.summary.scalar('n-step_reward', rew, step)

    def _calculate_reward(self, obs, metadata={}):
        throughput = obs[1]
        replicas = obs[3]
        avg_cpu = obs[4]
        avg_mem = obs[5]

        alpha = 0.75
        beta = 0.125
        gamma = 0.125
        phi = 0.25
        
        r_th = alpha * (throughput ** 2)
        r_cpu = beta * (avg_cpu*100)
        r_mem = gamma * (avg_mem*100)
        r_rep = -phi * ((replicas - self.MIN_PODS) ** 2)

        reward = round(r_th + r_cpu + r_mem + r_rep, 2)

        if (metadata['scale_value'] != replicas):
            reward += self.reward_range[0]
        
        return reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0
        self.loop = 0
        observation = self._get_obs()
        info = self._get_info()
        self._last_obs = observation
        return observation, info

    def step(self, action):
        done = False
        action = self._action_to_scale[action]
        info = self._take_action(action=action)

        if info['action_feedback'] == False:
            self._write_to_board(self._last_obs, action, -100, info, self.timestep, self.episode)
            self.timestep += 1
            self.loop += 1
            self.score += -100
            # End episode if 10 invalid actions in a row? (Original logic preserved)
            if self.loop == 10:
                done = True
            return self._last_obs, -100, done, False, info 
        else:
            time.sleep(self.sampling_window)
            next_obs = self._get_obs()
            reward = self._calculate_reward(obs=next_obs, metadata=info)
            self.score += round(reward, 2)
            self._write_to_board(next_obs, action, reward, info, self.timestep, self.episode)

            self.timestep += 1
            # End episode every 10 timesteps
            if (self.timestep % 10 == 0):
                done = True
                self.episode += 1
                self.loop = 0
                self.reward_history.append(self.score)
                with self.file_writer.as_default():
                    tf.summary.scalar('episodic_reward', self.score, self.episode)
                
                self.score = 0
                history = {'reward_history': self.reward_history, 'last_episode': self.episode}
                with open(self._reward_file, "w") as outfile:
                    json.dump(history, outfile)
                
            self._last_obs = next_obs
            return next_obs, reward, done, False, info

    def render(self, mode='human', close=False): pass
    def close(self): pass