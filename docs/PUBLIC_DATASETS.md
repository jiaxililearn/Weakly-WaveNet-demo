# Public Datasets for Spatiotemporal Graph Anomaly Detection

Based on your model architecture (GWNet + Graph-level attention for anomaly detection), here are suitable public datasets:

## ✅ **BEST MATCHES for Your Architecture**

### 1. **METR-LA / PEMS-BAY** (Already in your repo!)
**What it is:** Traffic speed/flow from highway sensors in LA/Bay Area
- **Nodes:** 207 (METR-LA), 325 (PEMS-BAY)
- **Features:** Traffic speed, flow
- **Temporal:** 5-minute intervals
- **Duration:** Several months
- **For anomaly detection:** Detect traffic incidents, accidents, unusual congestion
  
**How to use:**
```python
# You already have METR-LA in data/METR-LA/
# Create anomaly labels from extreme events:
# - Label windows with speed drops >50% as anomalies
# - Label rush hour anomalies (speed < threshold during off-peak)
# - Label from known incident timestamps (if available)
```

**Pros:** 
- ✅ Already downloaded
- ✅ Well-studied baseline
- ✅ Clear spatiotemporal structure
- ✅ Easy to create synthetic anomaly labels

**Cons:**
- ❌ No ground-truth anomaly labels (you create them)

---

### 2. **NYC Taxi / Bike-sharing Data** 
**What it is:** Ride demand across city zones
- **Source:** [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), [Citi Bike](https://citibikenyc.com/system-data)
- **Nodes:** 263 taxi zones or ~800 bike stations
- **Features:** Pickup/dropoff counts, duration
- **Temporal:** Hourly aggregation
- **For anomaly detection:** 
  - Unusual demand spikes (events, emergencies)
  - Service disruptions
  - Weather anomalies
  - Major events (protests, parades)

**How to get labels:**
- Match with NYC event calendar
- Weather anomalies (blizzards, hurricanes)
- Known service outages

---

### 3. **SWaT (Secure Water Treatment)** ⭐ RECOMMENDED
**What it is:** Industrial control system with labeled cyberattacks
- **Source:** [iTrust Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
- **Nodes:** 51 sensors/actuators in water treatment plant
- **Features:** Water levels, flow rates, pump status
- **Temporal:** 1-second intervals
- **Anomalies:** 36 real cyberattacks with ground truth labels
- **Duration:** 11 days (7 normal + 4 attack)

**Why perfect for you:**
- ✅ **Real anomaly labels** (not synthetic)
- ✅ Spatiotemporal dependencies (water flows through system)
- ✅ Graph structure (connected processes)
- ✅ Challenging (attack patterns are subtle)
- ✅ Benchmark for anomaly detection papers

**Access:** Need to request from iTrust (academic use)

---

### 4. **WADI (Water Distribution)** 
**What it is:** Similar to SWaT, larger water distribution system
- **Source:** [iTrust Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
- **Nodes:** 127 sensors/actuators
- **Features:** Flow, pressure, levels
- **Anomalies:** 15 attack scenarios
- **Duration:** 16 days

**Same benefits as SWaT, larger scale**

---

### 5. **SMD (Server Machine Dataset)** ⭐ EASY ACCESS
**What it is:** Server metrics from a large internet company
- **Source:** [GitHub - SMD](https://github.com/NetManAIOps/OmniAnomaly)
- **Nodes:** 28 machines
- **Features:** 38 metrics per machine (CPU, memory, network, etc.)
- **Temporal:** 1-minute intervals
- **Anomalies:** Labeled server failures, performance issues
- **Duration:** 5 weeks per machine

**Why good for you:**
- ✅ **Real labels** from production incidents
- ✅ Public & easy download
- ✅ Multiple machines = graph nodes
- ✅ Well-studied baseline
- ✅ Clear spatiotemporal patterns

**Graph construction:** Machines communicate → edges based on network traffic/dependencies

---

### 6. **SMAP / MSL (NASA Spacecraft)** 
**What it is:** Satellite telemetry with anomalies
- **Source:** [NASA](https://github.com/khundman/telemanom)
- **Nodes:** 55 spacecraft subsystems
- **Features:** Temperature, voltage, current sensors
- **Anomalies:** Real spacecraft anomalies labeled by experts
- **Duration:** Varies by mission

**Why interesting:**
- ✅ Real expert-labeled anomalies
- ✅ High-stakes domain (spacecraft!)
- ✅ Subsystems have dependencies → graph structure

---

### 7. **UCR Time Series Anomaly Archive**
**What it is:** Collection of labeled time series anomalies
- **Source:** [UCR Archive](https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/)
- **Multiple datasets:** ECG, power demand, space shuttle, etc.
- **Labels:** Point anomalies marked by experts

**Note:** Mostly **univariate** - need to adapt for your graph model

---

### 8. **AIOps Challenge Datasets** (KPI Anomaly Detection)
**What it is:** Real-world KPIs from tech companies
- **Source:** [AIOps Challenge](http://iops.ai/competition_detail/?competition_id=5)
- **Nodes:** Multiple microservices
- **Features:** Response time, error rate, throughput
- **Anomalies:** Service failures, degradation

---

### 9. **Air Quality Data** (EPA / OpenAQ)
**What it is:** Pollution sensors across cities
- **Source:** [OpenAQ](https://openaq.org/), [EPA AQS](https://www.epa.gov/aqs)
- **Nodes:** Monitoring stations (spatial graph)
- **Features:** PM2.5, PM10, O3, NO2, etc.
- **Anomalies:** 
  - Wildfires (pollution spikes)
  - Industrial accidents
  - Sensor malfunctions

**How to use:**
- Match with wildfire databases
- Extreme value detection
- Cross-reference news events

---

### 10. **COVID-19 Mobility Data** (Google, Apple)
**What it is:** Movement patterns during pandemic
- **Source:** [Google Mobility Reports](https://www.google.com/covid19/mobility/)
- **Nodes:** Geographic regions
- **Features:** Mobility trends by category
- **Anomalies:** Lockdowns, events, policy changes

---

## 🎯 **MY TOP 3 RECOMMENDATIONS FOR YOU:**

### **1st Choice: SMD (Server Machine Dataset)**
**Why:** 
- ✅ Publicly available (no approval needed)
- ✅ Real anomaly labels
- ✅ Good data quality
- ✅ ~28 nodes fits your model size
- ✅ Many papers use it (easy comparison)
- ✅ Download link: https://github.com/NetManAIOps/OmniAnomaly

**Setup:**
```python
# Treat each machine as a graph node
# Features: 38 KPIs per machine
# Edges: Based on service dependencies or correlation
# Labels: Binary (normal/anomaly) at time-window level
```

### **2nd Choice: SWaT (Water Treatment)**
**Why:**
- ✅ Best quality labels (real attacks by security experts)
- ✅ True spatiotemporal graph (water flows)
- ✅ Challenging but fair benchmark
- ✅ 51 nodes (good size for your architecture)

**Caveat:** Need to request access (1-2 weeks approval)

### **3rd Choice: METR-LA (Already have it!)**
**Why:**
- ✅ You already have it!
- ✅ Create labels from extreme events
- ✅ Large-scale (207 nodes)
- ✅ Well-understood domain

**How to create anomaly labels:**
```python
# Strategy 1: Statistical anomalies
speed_mean = data.mean()
speed_std = data.std()
anomaly = (data < speed_mean - 3*speed_std) | (data > speed_mean + 3*speed_std)
anomaly_windows = aggregate_to_windows(anomaly)

# Strategy 2: Rush hour anomalies
normal_rush_hour_speed = get_rush_hour_baseline()
anomaly = (is_rush_hour & speed > normal_rush_hour_speed * 1.5) | \
          (is_off_peak & speed < normal_off_peak_speed * 0.5)

# Strategy 3: Use incident reports (if available from LA DOT)
```

---

## 📝 **Quick Comparison Table**

| Dataset | Nodes | Labels | Access | Domain | Quality |
|---------|-------|--------|--------|--------|---------|
| **SMD** | 28 | ✅ Real | 🟢 Public | Servers | ⭐⭐⭐⭐⭐ |
| **SWaT** | 51 | ✅ Real | 🟡 Request | Water | ⭐⭐⭐⭐⭐ |
| **WADI** | 127 | ✅ Real | 🟡 Request | Water | ⭐⭐⭐⭐⭐ |
| **METR-LA** | 207 | ❌ Create | 🟢 Have it | Traffic | ⭐⭐⭐⭐ |
| **SMAP** | 55 | ✅ Real | 🟢 Public | Space | ⭐⭐⭐⭐ |
| **MSL** | 27 | ✅ Real | 🟢 Public | Space | ⭐⭐⭐⭐ |
| **NYC Taxi** | 263 | ❌ Create | 🟢 Public | Transport | ⭐⭐⭐ |

---

## 🚀 **Getting Started with SMD (Easiest)**

```bash
# 1. Clone the repo
git clone https://github.com/NetManAIOps/OmniAnomaly.git
cd OmniAnomaly

# 2. Download SMD data
# Follow their instructions - data is in ServerMachineDataset/

# 3. Convert to your format
# Each machine is a node
# 38 features per machine
# Time windows: aggregate to ~10 min windows
# Graph: Connect machines based on correlation or known dependencies
```

---

## 📊 **Data Preparation Script Template**

I can create a script to convert any of these to your format:
- Input: `(num_samples, seq_length, num_nodes, features)`
- Labels: Binary (0=normal, 1=anomaly) per sample
- Graph: Adjacency matrix or edge list

Which dataset would you like me to help you prepare?
