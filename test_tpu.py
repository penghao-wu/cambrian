import urllib3.util.retry

# Save the original is_exhausted method
original_is_exhausted = urllib3.util.retry.Retry.is_exhausted

# Define a new is_exhausted method that always returns False
def infinite_is_exhausted(self):
    return False

# Monkey-patch the is_exhausted method
urllib3.util.retry.Retry.is_exhausted = infinite_is_exhausted
original_increment = urllib3.util.retry.Retry.increment

# Define a new increment method that does not decrement counts
def infinite_increment(self, method=None, url=None, response=None, error=None,
                       _pool=None, _stacktrace=None):
    # Keep the retry counts unchanged
    total = self.total
    connect = self.connect
    read = self.read
    redirect = self.redirect
    status_count = self.status

    # Rest of the original increment logic, without decrementing counts
    cause = 'unknown'
    status = None
    redirect_location = None

    if error and self._is_connection_error(error):
        pass  # Do not decrement connect retries
    elif error and self._is_read_error(error):
        pass  # Do not decrement read retries
    elif response and response.get_redirect_location():
        redirect_location = response.get_redirect_location()
        status = response.status
    else:
        cause = urllib3.exceptions.ResponseError.GENERIC_ERROR
        if response and response.status:
            status = response.status

    history = self.history + (
        urllib3.util.retry.RequestHistory(method, url, error, status, redirect_location),
    )

    new_retry = self.new(
        total=total,
        connect=connect,
        read=read,
        redirect=redirect,
        status=status_count,
        history=history,
    )

    # Do not raise MaxRetryError
    return new_retry

# Monkey-patch the increment method
urllib3.util.retry.Retry.increment = infinite_increment
import requests
from requests.adapters import HTTPAdapter

# Create a retry strategy with any initial counts (they won't decrement)
retry_strategy = urllib3.util.retry.Retry(
    total=None,  # Can be None or any number since is_exhausted returns False
    backoff_factor=1,  # Delay factor between retries
    status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # HTTP methods to retry
)

# Create an HTTP adapter with the retry strategy
adapter = HTTPAdapter(max_retries=retry_strategy)

# Create a session and mount the adapter
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

# Now use the session to make requests
try:
    response = session.get('http://example.com')
    # Process the response as needed
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
import torch_xla._internal.tpu as tpu_module
x = tpu_module.get_tpu_env()


# "ACCELERATOR_TYPE: 'v4-256'\nAGENT_BOOTSTRAP_IMAGE: 'gcr.io/cloud-tpu-v2-images/instance_agent:stable'\nALT: 'false'\nCHIPS_PER_HOST_BOUNDS: '2,2,1'\nCOLLECTD_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/collectd-agent:cl_225018473'\nCONSUMER_PROJECT_ID: 'nyu-vision-lab'\nCONSUMER_PROJECT_NUMBER: '373177222751'\nENABLE_ICI_RESILIENCY: 'true'\nENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY: 'false'\nENABLE_MEMCACHED: 'false'\nFLUENTD_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/fluentd-agent:cl_547826204'\nFORWARD_LIBTPU_LOGS: 'true'\nHEALTH_AGENT_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/tpu_agents:cl_684025707'\nHOST_BOUNDS: '2,2,8'\nINFERENCE_MODE: 'false'\nINJECT_SLICE_BUILDER_FAULT: ''\nINTERNAL: 'false'\nMAINTENANCE_ACTION_FLAG: 'unhealthy-maintenance'\nMEMCACHED_DOCKER_URL: ''\nMONITORING_AGENT_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/monitoring_agent:cloud_tpu.monitoringagent.1vm_20240925_RC00'\nNODE_ID: 'penghao-tpu-1'\nPREEMPTIBLE: 'false'\nRUNTIME_MONITOR_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/runtime_monitor:cl_534554121'\nRUNTIME_VERSION: 'gcr.io/cloud-tpu-v2-images/fake_tensorflow:latest'\nRUNTIME_VERSION_CHANGER_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/tf_version_changer:cl_302107686'\nSERVICE_NAME: 'tpu.googleapis.com'\nSOURCE: 'TFRC'\nTOPOLOGY: '4x4x8'\nTPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1'\nTPU_PROCESS_BOUNDS: '2,2,8'\nTPU_RUNTIME_METRICS_PORTS: '8431,8432,8433,8434'\nTPU_TOPOLOGY_ALT: 'false'\nTPU_TOPOLOGY_WRAP: 'true,true,true'\nTYPE: 'V4'\nUID: '6502082742571311568'\nUSE_DIRECT_PATH: 'false'\nVBARCONTROL_AGENT_DOCKER_URL: ''\nWORKER_ID: '0'\nWRAP: 'true,true,true'\nZONE: 'us-central2-b'\n"

# TPU_SKIP_MDS_QUERY='True'
# TPU_ACCELERATOR_TYPE='v4-256'
# TPU_PROCESS_BOUNDS='2,2,8'
# TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1'
# TPU_WORKER_ID=''