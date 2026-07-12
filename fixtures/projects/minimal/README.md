# Minimal synthetic project

This fixture contains only small, committed synthetic inputs. Every record and
event identity is reproducible from the adjacent canonical JSON bytes. The
remaining rendering-contract identity is the SHA-256 of the exact committed
bytes in `sources/rendering-contract.txt`.

From the repository root:

```text
temper status fixtures/projects/minimal
temper verify fixtures/projects/minimal
temper dump fixtures/projects/minimal
temper manifest fixtures/projects/minimal --type project --id project-minimal
```

Derived state is deliberately absent. It is rebuildable from the verified
event stream and is not canonical evidence.
