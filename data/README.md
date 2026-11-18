# Data

The corpora are not redistributed here. Download them from the Canadian
Institute for Cybersecurity and place them as follows.

## CIC IoT-DIAD 2024 (main corpus)

Source: https://www.unb.ca/cic/datasets/iot-diad-2024.html

Download the packet-based CSV files and put the 28 files below in
`data/cic_iot_diad_2024/` (file names exactly as published):

    BenignTraffic1  DDoS-ACK_Fragmentation12  DDoS-HTTP_Flood-  DDoS-ICMP_Fragmentation
    DDoS-SlowLoris  DDoS-SynonymousIP_Flood  DDoS-TCP_Flood  DDoS-UDP_Flood
    DDoS-UDP_Fragmentation  DoS-HTTP_Flood  DoS-SYN_Flood  DoS-TCP_Flood  DoS-UDP_Flood
    Recon-HostDiscovery  Recon-OSScan  Recon-PingSweep  Recon-PortScan  VulnerabilityScan
    Mirai-greip_flood1  DictionaryBruteForce  MITM-ArpSpoofing  DNS_Spoofing
    BrowserHijacking  CommandInjection  SqlInjection  XSS  Uploading_Attack  Backdoor_Malware

Then, from the repository root:

    bash src/journal/extract_device_class_counts.sh   # per-device class counts (optional:
                                                       # a copy is in results/journal/)
    python -m src.journal.dataset_builder              # -> data/diad_federated/

`dataset_builder` keeps `device_mac` only as the partition key, drops identifier
columns, samples uniformly within each class (cap 150,000, seed 42) and leaves
imputation and variance filtering to the training split of each run.

## CICIoT2023 (transfer corpus)

Source: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Put `part-00000-ciciot2023.csv` (the first part of the published CSV release)
in `data/ciciot2023/`, then:

    python -m src.journal.preprocessing --dataset ciciot2023   # -> data/clean_ciciot2023/

The scaler is fitted on the training split only.

## Checksums

Every result file records the SHA-256 of the processed arrays in its
`_manifest` block, so a run on a different download can be told apart.
