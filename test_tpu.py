import torch_xla._internal.tpu as tpu_module
x = tpu_module.get_tpu_env()


# "ACCELERATOR_TYPE: 'v4-256'\nAGENT_BOOTSTRAP_IMAGE: 'gcr.io/cloud-tpu-v2-images/instance_agent:stable'\nALT: 'false'\nCHIPS_PER_HOST_BOUNDS: '2,2,1'\nCOLLECTD_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/collectd-agent:cl_225018473'\nCONSUMER_PROJECT_ID: 'nyu-vision-lab'\nCONSUMER_PROJECT_NUMBER: '373177222751'\nENABLE_ICI_RESILIENCY: 'true'\nENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY: 'false'\nENABLE_MEMCACHED: 'false'\nFLUENTD_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/fluentd-agent:cl_547826204'\nFORWARD_LIBTPU_LOGS: 'true'\nHEALTH_AGENT_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/tpu_agents:cl_684025707'\nHOST_BOUNDS: '2,2,8'\nINFERENCE_MODE: 'false'\nINJECT_SLICE_BUILDER_FAULT: ''\nINTERNAL: 'false'\nMAINTENANCE_ACTION_FLAG: 'unhealthy-maintenance'\nMEMCACHED_DOCKER_URL: ''\nMONITORING_AGENT_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/monitoring_agent:cloud_tpu.monitoringagent.1vm_20240925_RC00'\nNODE_ID: 'penghao-tpu-1'\nPREEMPTIBLE: 'false'\nRUNTIME_MONITOR_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/runtime_monitor:cl_534554121'\nRUNTIME_VERSION: 'gcr.io/cloud-tpu-v2-images/fake_tensorflow:latest'\nRUNTIME_VERSION_CHANGER_DOCKER_URL: 'gcr.io/cloud-tpu-v2-images/tf_version_changer:cl_302107686'\nSERVICE_NAME: 'tpu.googleapis.com'\nSOURCE: 'TFRC'\nTOPOLOGY: '4x4x8'\nTPU_CHIPS_PER_PROCESS_BOUNDS: '2,2,1'\nTPU_PROCESS_BOUNDS: '2,2,8'\nTPU_RUNTIME_METRICS_PORTS: '8431,8432,8433,8434'\nTPU_TOPOLOGY_ALT: 'false'\nTPU_TOPOLOGY_WRAP: 'true,true,true'\nTYPE: 'V4'\nUID: '6502082742571311568'\nUSE_DIRECT_PATH: 'false'\nVBARCONTROL_AGENT_DOCKER_URL: ''\nWORKER_ID: '0'\nWRAP: 'true,true,true'\nZONE: 'us-central2-b'\n"

# TPU_SKIP_MDS_QUERY='True'
# TPU_ACCELERATOR_TYPE='v4-256'
# TPU_PROCESS_BOUNDS='2,2,8'
# TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1'
# TPU_WORKER_ID=''