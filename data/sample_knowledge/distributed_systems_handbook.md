# Distributed Systems & Consensus Engineering Handbook

## Consensus Protocols: Raft vs Multi-Paxos
Distributed consensus allows a cluster of nodes to agree on a sequence of state machine transitions even in the presence of network partitions or machine crashes.

### Raft Consensus Mechanics
Raft decomposes consensus into three distinct subproblems:
1. **Leader Election**: When a follower's election timer expires (randomized between 150ms and 300ms), it transitions to candidate state, increments its current term, and broadcasts `RequestVote` RPCs. A candidate becomes leader upon receiving votes from a majority of nodes (quorum = `floor(N/2) + 1`).
2. **Log Replication**: The leader receives client commands, appends them to its log, and sends `AppendEntries` RPCs to all followers. An entry is considered committed once replicated on a majority of cluster nodes.
3. **Safety**: Raft guarantees that if a leader has committed a log entry for a given term, that entry will be present in the logs of the leaders for all higher terms.

### Byzantine Fault Tolerance (BFT)
Unlike crash-fault tolerant algorithms (Raft, Paxos) that assume nodes are honest and merely fail by stopping, Byzantine Fault Tolerant systems (like PBFT and Tendermint) tolerate malicious, colluding, or arbitrary node behaviors. PBFT requires $3f + 1$ total nodes to tolerate $f$ Byzantine adversarial nodes.

## The CAP Theorem in Practice
Eric Brewer's CAP Theorem states that in any distributed data store, you can only guarantee two out of the three properties:
- **Consistency (Linearizability)**: Every read receives the most recent write or an error.
- **Availability**: Every non-failing node returns a non-error response for every request.
- **Partition Tolerance**: The system continues to operate despite arbitrary message loss or network partitions.

Since network partitions are inevitable in real-world physical networks, distributed system architects must choose between CP (Consistency over Availability, e.g. CockroachDB, etcd) and AP (Availability over Consistency, e.g. Cassandra, DynamoDB).
