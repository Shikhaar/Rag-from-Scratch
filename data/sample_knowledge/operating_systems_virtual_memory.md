# Operating Systems: Virtual Memory & Paging Subsystems

## 1. Multi-Level Page Tables & TLB Architecture
Modern 64-bit architectures (such as x86-64 4-level paging: PML4, PDPT, PD, PT) map virtual memory addresses to physical frames. 

To mitigate the $O(4)$ memory access penalty per address translation, the CPU utilizes a Translation Lookaside Buffer (TLB):
- **TLB Hit**: Address translation resolved in a single CPU cycle.
- **TLB Miss**: Hardware page table walker traverses the page hierarchy in main RAM.
- **TLB Shootdown**: When a page table entry is modified on one core (e.g., during `munmap` or page migration), an Inter-Processor Interrupt (IPI) must be broadcast to flush stale TLB entries across all other CPU cores.

## 2. Page Fault Handling & Demand Paging
When a thread accesses a virtual address not mapped in the hardware page table, the CPU triggers a page fault exception (`#PF` vector 14):
1. The kernel page fault handler inspects the `CR2` control register containing the faulting virtual address.
2. The kernel verifies whether the address lies within a valid Virtual Memory Area (`vm_area_struct`).
3. If valid but unallocated, a physical frame is allocated from the buddy allocator and zeroed (demand zero-paging).
4. If the page is swapped to disk, an asynchronous disk read is dispatched to load the page back into RAM.
5. The Page Table Entry (PTE) is populated with Present=1, and the faulting instruction is restarted.

## 3. HugePages and Transparent HugePages (THP)
Standard x86 pages are 4KB. Enterprise databases and LLM inference engines utilize 2MB HugePages or 1GB Pages to drastically reduce TLB footprint and eliminate TLB miss thrashing across multi-gigabyte memory buffers.
