
---

### 1. **Flow\_Timing**

These features describe **timing characteristics** of network flows—how long flows last and the time intervals between packets.

* **Flow Duration**: Total time from the first packet to the last packet in the flow.
* **Flow IAT Mean/Std/Max/Min**: Inter-Arrival Time (IAT) between packets in the entire flow — average, standard deviation, maximum, minimum.
* **Fwd IAT Total/Mean/Std/Max/Min**: Same as above but only for packets in the **forward direction** (usually client → server).
* **Bwd IAT Total/Mean/Std/Max/Min**: Same but for packets in the **backward direction** (server → client).

---

### 2. **Packet\_Size**

These describe **packet size characteristics** within the flow.

* **Total Length of Fwd/Bwd Packets**: Sum of bytes sent forward/backward.
* **Fwd/Bwd Packet Length Max/Min/Mean/Std**: Max, min, average, and standard deviation of packet lengths in forward and backward directions.
* **Min Packet Length / Max Packet Length**: Across all packets in the flow.
* **Packet Length Mean/Std/Variance**: Statistics for packet sizes across the flow.
* **Average Packet Size**: Mean size of packets.
* **Avg Fwd Segment Size / Avg Bwd Segment Size**: Average segment size in each direction, often related to TCP segment size.

---

### 3. **Flow\_Rates**

Rate features describe how **fast bytes or packets are transmitted** over the flow duration.

* **Flow Bytes/s**: Total bytes in flow divided by flow duration (bytes per second).
* **Flow Packets/s**: Total packets per second.
* **Fwd Packets/s** and **Bwd Packets/s**: Packet rates in each direction.

---

### 4. **TCP\_Flags**

These features count how many times **TCP flags** occur in the flow's packets. TCP flags signal specific control information:

* **FIN, SYN, RST, PSH, ACK, URG**: Common TCP control flags used for connection management and signaling.
* **CWE, ECE**: Congestion control flags.
* **Fwd PSH Flags / Bwd PSH Flags**: Count of PSH flags in forward/backward packets (indicating urgent data pushing).
* **Fwd URG Flags / Bwd URG Flags**: Count of URG flags in each direction (urgent data).

---

### 5. **Protocol\_Network**

Features related to the **network and transport protocol specifics**.

* **Source Port / Destination Port**: Ports involved in the connection (e.g., 80 for HTTP).
* **Protocol**: Protocol number (TCP=6, UDP=17, etc.).
* **Fwd Header Length / Bwd Header Length**: Size of headers in each direction (could include TCP/IP header length).

---

### 6. **Packet\_Counts**

Counts of packets and bytes to capture flow volume characteristics.

* **Total Fwd Packets / Total Backward Packets**: Number of packets sent forward and backward.
* **Subflow Fwd/Bwd Packets and Bytes**: Packets and bytes in subflows or smaller chunks within the main flow (often used in TCP session analysis).

---

### 7. **Activity\_State**

Describe **periods of activity and idle times** within the flow.

* **Active Mean/Std/Max/Min**: Statistics of active times (periods when packets are being transmitted).
* **Idle Mean/Std/Max/Min**: Statistics of idle times (periods of no packet transmission).

---

### 8. **TCP\_Connection**

Features related to the **TCP connection window and packet sizes**.

* **Init\_Win\_bytes\_forward / backward**: Initial TCP window size (bytes) advertised by each side.
* **act\_data\_pkt\_fwd**: Number of packets with actual data in the forward direction.
* **min\_seg\_size\_forward**: Minimum TCP segment size seen in forward direction.

---

### 9. **Bulk\_Transfer**

Features describing characteristics of **bulk data transfer** within the flow.

* **Fwd/Bwd Avg Bytes/Bulk**: Average bytes per bulk transfer in forward/backward direction.
* **Fwd/Bwd Avg Packets/Bulk**: Average packets per bulk transfer.
* **Fwd/Bwd Avg Bulk Rate**: Average rate of bulk transfer (bytes or packets per time unit).

---

### 10. **Direction\_Analysis**

Analyzes **directional properties** of the flow.

* **Down/Up Ratio**: Ratio of bytes or packets sent downward (usually from server to client) vs upward.
* **Inbound**: Binary or numerical indicator if the flow is inbound (to the monitored network) or outbound.

---

### 11. **Special\_Features**

Custom or domain-specific features.

* **SimillarHTTP**: Possibly a measure or flag indicating similarity of this flow to typical HTTP traffic patterns (e.g., based on HTTP headers or payload).

---

### Summary:

* **Flow-level features** summarize an entire communication session (flow) between two endpoints.
* They capture **timing, size, count, rates, TCP control signals, directionality, and protocol info**.
* These features are often used in **network intrusion detection systems (NIDS), traffic classification, anomaly detection**, and network performance monitoring.

---