# Distributed Event Streaming & Consensus: Apache Kafka Internals

## 1. Partition Log Architecture & In-Sync Replicas (ISR)
A Kafka topic is partitioned into immutable append-only commit logs. Each partition has one Leader broker and multiple Follower replicas:
- **In-Sync Replicas (ISR)**: The subset of replicas that are actively caught up with the partition leader's log end offset (`LEO`).
- **High Watermark (HW)**: The offset up to which all ISR replicas have acknowledged log replication. Consumers can only read messages below the High Watermark.
- **`min.insync.replicas`**: If the number of active ISR members falls below this configured threshold, producer writes with `acks=all` are rejected with `NotEnoughReplicasException`.

## 2. Consumer Group Rebalancing Protocol
Consumers within a group coordinate partition assignments via the Group Coordinator broker and Group Leader:
1. **Heartbeat Thread**: Sends periodic heartbeats (`heartbeat.interval.ms`) to the coordinator.
2. **Rebalance Triggers**: Consumer failure, consumer addition, or partition scaling triggers a rebalance.
3. **Cooperative Sticky Assignor**: Rebalances partitions incrementally without pausing processing across unchanged partition assignments (eliminating the "stop-the-world" effect of legacy Eager rebalancing).

## 3. Exactly-Once Processing (EOS) Semantics
Kafka achieves EOS via two core primitives:
- **Idempotent Producer**: Attaches a unique Producer ID (`PID`) and monotonically increasing Sequence Number to each batch, allowing broker deduping on network retries.
- **Transactional Coordinator**: Uses a two-phase commit over the `__transaction_state` internal topic to atomically write messages across multiple topic-partitions alongside consumer offset commits.
