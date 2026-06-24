# Scaling Agent Testing Framework


### Linux Server Setup

The following setup is for running the real OpenFaaS experiment harness on a Linux server or VM. It assumes Ubuntu/Debian, a user with `sudo`, and a server that can run Kubernetes, OpenFaaS, Prometheus, the state tracker, and VisuFaaS.

1.  Install base packages:

```bash
sudo apt update
sudo apt install -y git curl python3 python3-venv python3-pip docker.io
sudo usermod -aG docker "$USER"
```

Log out and back in after adding the user to the `docker` group.

2.  Install MicroK8s and prepare `kubectl` access:

```bash
sudo snap install microk8s --classic --channel=1.27/stable
sudo usermod -aG microk8s "$USER"
mkdir -p ~/.kube
microk8s config > ~/.kube/config
chmod 600 ~/.kube/config
microk8s enable dns storage metrics-server
```

3.  Install the OpenFaaS CLI and deploy OpenFaaS using the official OpenFaaS deployment/workshop instructions:

```bash
curl -sLS https://cli.openfaas.com | sudo sh
kubectl get nodes
kubectl get pods -n openfaas
kubectl get pods -n openfaas-fn
```

Expose the services expected by this project. The default tracker configuration expects OpenFaaS on port `8080` and Prometheus on port `9090`:

```bash
kubectl -n openfaas port-forward svc/gateway 8080:8080
kubectl -n openfaas port-forward svc/prometheus 9090:9090
```

Run the port-forwards in separate terminal sessions, or replace them with your server's service exposure method.

4.  Deploy the matrix multiplication OpenFaaS function from `openfaas-functions/`. The state tracker defaults to `FUNCTION_NAME=matmul` and `FUNCTION_NAMESPACE=openfaas-fn`, so keep the function name aligned with those variables:

```bash
faas-cli list
faas-cli deploy -f <function-stack-file>.yml
faas-cli invoke matmul
```

If OpenFaaS Community Edition limits the function to 5 replicas, update the `DefaultMaxReplicas` value in both `gateway` and `faas-netes` as described in [openfaas-functions/README.md](./openfaas-functions/README.md) before running high-replica experiments.

5.  Run [VisuFaaS](https://github.com/atasoya/visufaas) on the same server. VisuFaaS is required because the state tracker applies scale decisions through `GET http://127.0.0.1:3000/scale?replicas=<n>`.

```bash
git clone https://github.com/atasoya/visufaas.git
cd visufaas
```

Follow the VisuFaaS README for the current build/run command. Its default conventions match this project: OpenFaaS on `8080`, Prometheus on `9090`, and the VisuFaaS API on `3000`.

Verify the API before starting the tracker:

```bash
curl "http://127.0.0.1:3000/replicas"
curl "http://127.0.0.1:3000/scale?replicas=1"
```

6.  Start the NSGD state tracker:

```bash
cd openfaas-functions/state-tracker
python3 -m venv .venv
. .venv/bin/activate
pip install fastapi uvicorn httpx kubernetes prometheus-api-client requests pandas numpy scipy matplotlib tqdm flask

export OPENFAAS_GATEWAY="http://127.0.0.1:8080"
export PROMETHEUS_URL="http://127.0.0.1:9090"
export AGENT_URL="http://127.0.0.1:5000"
export FUNCTION_NAME="matmul"
export FUNCTION_NAMESPACE="openfaas-fn"
export MAX_REPLICAS="24"
export TRACKER_PORT="8000"
python app.py
```

Send workload traffic through the tracker rather than directly to OpenFaaS:

```bash
curl -X POST "http://127.0.0.1:8000/function/matmul"
python test_tracker.py
```

Useful tracker endpoints are `GET /health`, `GET /states/current`, `GET /states/summary`, `GET /metrics/latest`, and `GET /metrics/summary`.

For LSTM-PPO evaluation runs, start the passive tracker instead:

```bash
export OPENFAAS_GATEWAY="http://127.0.0.1:8080"
export PROMETHEUS_URL="http://127.0.0.1:9090"
export FUNCTION_NAME="matmul"
export FUNCTION_NAMESPACE="openfaas-fn"
export MAX_REPLICAS="24"
export MIN_REPLICAS="1"
export TRACKER_PORT="8000"
python rl_app.py
```

`rl_app.py` proxies traffic and writes the same CSV schema as the NSGD tracker, but it does not call the NSGD agent or apply scaling actions. The LSTM-PPO agent should run separately and scale through Kubernetes directly.

### State Tracker Contributions

The state tracker code in [openfaas-functions/state-tracker](./openfaas-functions/state-tracker/) is our real-system bridge between the simulator-oriented autoscaling algorithms and the OpenFaaS deployment used for experiments.

- `app.py` implements the active NSGD middleware. It proxies `/function/{name}` requests to OpenFaaS, records an atomic pre-admission snapshot, asks the NSGD agent for `theta`, converts `theta_step` into a replica target, calls the VisuFaaS scale endpoint, forwards the request, and logs the observed outcome.
- `FunctionState` in `app.py` and `rl_app.py` converts runtime signals into the five NSGD state components: `cold`, `idle_on`, `busy`, `init_free`, and `init_reserved`.
- `InFlightCounter` tracks concurrent requests at the proxy boundary so the tracker can distinguish busy capacity from initializing capacity at request arrival time.
- `pod_watcher.py` maintains a Kubernetes watch cache for OpenFaaS function pods labeled `faas_function=<function_name>`. It classifies pods as ready/not-ready, ignores terminating pods, reconnects after watch interruptions, and serves cached counts without querying Kubernetes on every request.
- `exp_controller.py` implements real-system idle expiration. It converts the learned `theta_exp` into an idle timeout and scales down through VisuFaaS while accounting for Kubernetes scale-down lag.
- `agent_call_service.py` isolates NSGD REST-agent communication. If the agent is unavailable, proxy traffic still reaches OpenFaaS and the error is recorded.
- `experiment_metrics.py` writes comparable experiment logs to `samples.csv`, `request_log.csv`, and `monitor.csv`. It computes NSGD equation-7 cost, records latency and rejection data, samples Prometheus/Kubernetes metrics, and computes LSTM-PPO reward terms for shared notebook analysis.
- `rl_app.py` is the passive observer used for DRe-SCale/LSTM-PPO runs. It keeps the same proxy, state derivation, and CSV schema as the NSGD tracker while leaving scaling decisions to the external RL agent.
- `test_tracker.py` provides a quick integration check for health, derived states, single requests, concurrent requests, and history accumulation before running a full workload.

Together, these contributions provide a reproducible OpenFaaS experiment harness: workload traffic enters through one observable proxy, Kubernetes pod state is tracked continuously, scaling commands are sent through VisuFaaS, and NSGD and DRe-SCale/LSTM-PPO runs produce directly comparable logs.



### Scaling Agents Explored 
1.  [LSTM integrated Proximal Policy Optimisation](./scaling-agents/LSTM-PPO/) (LSTM-PPO/Recurrent-PPO)
2.  [vanilla Proximal Policy Optimisation](./scaling-agents/PPO/) (PPO)
3.  [Deep Recurrent Q-Network](./scaling-agents/DRQN/) (LSTM integrated DQN/DRQN)
4.  [LSTM integrated Soft-Actor-Critic](./scaling-agents/SAC(+LSTM)) (LSTM-SAC and SAC)

### Training and Evaluation
1.  [Experimental Results](/experimental-results/images/)
2.  [Analysis File](./experimental-results/analysisUtility.ipynb)

### Made By

- ATA ATASOY
- DENİZ ERKİN KASAPLI
- HÜSEYİN BORAN KUŞÇU
