# Database Concurrency Control: 2PL, MVCC, and Isolation Anomalies

## 1. Two-Phase Locking (2PL) & Strict 2PL
Pessimistic concurrency control enforces serializability by acquiring locks before data operations:
- **Growing Phase**: Transactions acquire Shared (S) locks for reads and Exclusive (X) locks for writes. No locks can be released.
- **Shrinking Phase**: Locks are released monotonically. No new locks can be acquired.
- **Strict 2PL (SS2PL)**: All Exclusive locks are held until transaction commit or abort, preventing cascading aborts and dirty reads.

## 2. Multi-Version Concurrency Control (MVCC)
PostgreSQL and modern relational engines implement MVCC to guarantee that "readers never block writers, and writers never block readers":
- Each tuple header contains `xmin` (creating transaction ID) and `xmax` (deleting or updating transaction ID).
- When a row is updated, the engine does not overwrite the existing tuple; instead, it appends a new tuple with the current transaction's `xmin` and sets `xmax` on the old tuple.
- Vacuum processes periodically reclaim dead tuple versions that are no longer visible to active transaction snapshots.

## 3. Concurrency Anomalies & Snapshot Isolation
Under Snapshot Isolation (SI), transactions observe a consistent snapshot taken at the start of the transaction:
- Prevents Dirty Reads, Non-Repeatable Reads, and Phantom Reads.
- **Write Skew Anomaly**: Two concurrent transactions read overlapping disjoint data sets, evaluate a shared constraint (e.g. at least one doctor on-call), and concurrently update separate rows, violating the global invariant.
- **Serializable Snapshot Isolation (SSI)**: Detects dangerous structures in the serialization dependency graph (`rw-antidependency cycles`) and aborts one transaction to guarantee true serializability without lock contention.
