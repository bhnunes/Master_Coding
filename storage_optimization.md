# High-Performance WSI Patching Pipeline Configuration

## Overview

This document defines the optimal storage layout and execution strategy for WSI patch extraction and HDF5 generation, based on measured disk performance and system constraints.

---

## Hardware Summary

- CPU: 4 cores
- Drives:
  - C: 150 GB (OS + environment)
  - D: 500 GB
  - F: 1 TB (fast)
  - G: 1 TB (WSI storage)

---

## Storage Roles

### G:\ (Source Storage)
- Store only:
  - Raw WSI files
  - Original annotations
- Avoid:
  - Writing outputs
  - Temporary files
  - Logs

---

### F:\ (Primary Workspace)
- Main high-performance working directory
- Store:
  - HDF5 outputs
  - Logs
  - Progress tracking (SQLite, manifests)
  - Temporary staging files

Suggested structure:
```
F:\patching_workspace\
    staging\
    hdf5_output\
    logs\
    progress\
```

---

### C:\ (System Drive)
- Store only:
  - Code
  - Python environment
  - Config files
- Avoid heavy I/O

---

### D:\ (Secondary Storage)
- Use for:
  - Completed HDF5 archives
  - Overflow storage
  - Backup

---

## Pipeline Architecture

### Data Flow

1. Read WSIs from G:\
2. (Optional) Stage batch to F:\staging\
3. Extract patches in memory
4. Send batches to writer
5. Write HDF5 to F:\hdf5_output\
6. Move completed outputs to D:\ if needed

---

## Parallel Processing Strategy

### Recommended Configuration

- 3 Worker Processes (patch extraction)
- 1 Writer Process (HDF5 serialization)

### Responsibilities

#### Workers
- Read WSI regions
- Apply preprocessing
- Generate patch batches
- Push to queue

#### Writer
- Receive batches
- Append to HDF5
- Flush in chunks
- Maintain single-writer integrity

---

## Important Constraints

- HDF5 must be written by a **single writer**
- Avoid concurrent writes to the same file
- Use batching to reduce I/O overhead

---

## Batch Processing Strategy

Use a rolling batch approach:

1. Copy N WSIs from G:\ to F:\staging\
2. Process batch entirely from F:\
3. Save HDF5 to F:\
4. Move finished HDF5 to D:\ if needed
5. Delete staged WSIs
6. Repeat

---

## Performance Principles

- Prefer **sequential writes** over random writes
- Keep **hot loop on F:\**
- Minimize reads from external/slow disks
- Avoid small-file explosion (use HDF5)

---

## Anti-Patterns (Avoid)

- Writing outputs to G:\
- Heavy writes on C:\
- Multiple HDF5 writers
- Millions of PNG patches
- Over-parallelization (4+ workers reading simultaneously)

---

## Final Configuration Summary

```
Input WSIs:      G:
Staging:         F:
HDF5 output:     F:
Archive:         D:
Code/env:        C:

Workers:         3
Writer:          1
```

---

## Notes

- Start with conservative parallelism and scale if stable
- Monitor disk usage on F:\ to avoid overflow
- Validate HDF5 integrity before archiving

---

End of document.
