# Changelog

## Unreleased
- Added structural-only `Rank9SelIndex::from_bytes_with_len` attachment and
  `WaveletMatrix::from_layers` for borrowed query layers without a synthesized
  serialization table. These preserve the original byte owners and construct
  no indexes. Full rank/select verification remains explicit in `validate_for`
  and `from_bytes_for_data`; framing, geometry and truncation checks remain on
  ordinary attachment. Tests distinguish semantic audit from typed attachment
  and check backing pointers and raw `NoIndex` queries. Rank9 attachment checks
  hint-array lengths and terminal block sentinels in constant time, preserving
  the format's padded-zero convention; individual query paths use fallible
  hint/word access and checked arithmetic without scanning rank contents.
- Wavelet matrices now use the minimum code width
  `max(1, bit_length(alphabet_size - 1))`, removing the redundant leading-zero
  plane at power-of-two cardinalities. CPU and GPU rank/select APIs explicitly
  treat values outside `0..alphabet_size` as absent, preventing those values
  from aliasing low codes; raw and persisted-index attachment reject
  noncanonical layer counts.
- Added layer-major CPU `WaveletMatrix::rank_batch`, `rank_batch_into`, and
  `rank_range_batch_into` APIs. Independent probes now overlap Rank9 memory
  misses, shared-range queries use one two-endpoint descent instead of two
  separate ranks, batches below eight probes retain the scalar path, and
  Rank9 data/directory loads are scheduled to overlap. Randomized parity tests
  cover scalar equivalence and boundary cases.
- Made `WaveletMatrix::access` explicitly share one rank1 query between its
  zero and one branches, preserving the rank0 identity while stabilizing hot
  Rank9 code generation across unrelated downstream monomorphizations.
- Added `GpuContext::static_batch_dispatch` and its typed `StaticDispatch`
  result for allocation-free direct launches when the logical batch length is
  host-known. Static and device-written dispatches now share one checked
  geometry planner covering logical/capacity bounds, actual device limits,
  two-dimensional envelopes, zero lengths, and flattened `u32` positions.
- Extended the experimental GPU pipeline seam with caller-owned fixed and
  device-length/indirect resident WaveletMatrix access, plus `GpuBitVector`
  rank1/select1 batches built directly from canonical raw bit-vector data.
  All structures and buffers share `GpuContext` provenance, dynamic kernels
  preserve unused capacity, and producer→query→consumer WGPU tests perform no
  intermediate host read. Fixed launches use least-area legal 2-D workgroup
  rectangles; device-written indirect launches use a constant-time tight cover
  inside a host-validated capacity envelope. The 65,536-group boundary launches
  32,768×2 rather than 65,535×2, while the envelope prevents flattened `u32`
  index wrap.
- Added device-resident `u32` buffers and `GpuWaveletMatrix` rank launch APIs
  that accept resident positions/values and write resident results without a
  host transfer or synchronization. The existing slice-based `rank_batch`
  remains the convenient upload/launch/readback path, while CubeCL query
  pipelines can now chain kernels and read back only their final output. A
  cloneable `GpuContext` safely shares typed buffers across multiple matrices
  while rejecting buffers from unrelated client domains. Dynamic resident
  batches use separate ordinary `[logical_len, capacity]` metadata and an
  exclusive persistent indirect-dispatch record; rank guards on the
  device-produced logical length and never probes unused capacity slots.
- Added zero-copy persistence and checked attachment for `Rank9SelIndex`, plus
  `WaveletMatrix` helpers that write and reattach one layer index at a time in
  MSB-to-LSB order. Checked attachment validates every rank/subrank and select
  hint against the raw bit-vector words without allocating an index-sized
  buffer, rejecting malformed or semantically incompatible sidecars. Wavelet
  reconstruction now also bounds-checks raw section handles before AnyBytes
  slicing, returning metadata errors instead of panicking on invalid ranges.
- Added an optional `gpu` feature with GPU batch query kernels for
  `WaveletMatrix` (`jerky::gpu::GpuWaveletMatrix`, cubecl/wgpu): upload the
  bit-planes plus a flat rank-block directory once, then answer batches of
  access/rank/select/quantile queries with one dispatch and one host↔device
  sync per batch. Results are bit-identical to the CPU API (property-tested
  to 4M values / 64k alphabets, including out-of-range queries). Measured on
  an M4 Max at 1M-query batches: 24-72x over one core, 2.2-4.9x over all 16
  threads; break-even vs 16 threads is roughly 16k-65k queries per batch
  (`examples/gpu_bench.rs`). Intended for large batch analytic workloads
  (the SuccinctArchive shape); point lookups should stay on the CPU form.
- Added a linear structural merge for `WaveletMatrix`: `merge_interleaved`
  (interleave N built matrices per a caller-supplied origin order by merging
  bit-planes directly — no decode, no sort, no freeze permutation) and
  `merge_sorted` (multi-way merge of matrices storing sorted runs — the LSMT
  segment-compaction primitive). Emits a complete queryable matrix
  (bit-planes + rank/select indexes); ~14x faster than decode+sort+rebuild at
  5M values (`bench/src/merge_vs_rebuild.rs`).
- Added `WaveletMatrix::to_vec`, a sequential-pass decoder (inverse freeze)
  for recovering the whole stored sequence, and
  `BitVectorBuilder::words_mut` for bulk word-level writes.
- Allowed `WaveletMatrix` builders to construct empty sequences, skipping layer
  permutations for zero-length inputs and covering the case with a regression
  test.
- Simplified `WaveletMatrixBuilder::freeze` to drain builders in the common
  path, removing redundant early returns while preserving empty-layer support.
- Added metadata validation in `BitVectorData::from_bytes` to reject lengths
  exceeding the stored capacity and covered the case with a regression test.
- Guarded `CompactVector::from_bytes` against metadata bit-length overflow.
- Removed the `anyhow` dependency in favor of crate-defined errors and updated
  `Serializable` to expose an associated error type for reconstruction.
- Prevent panic in `DacsByte::len` by handling empty level lists gracefully.
- Embedded section handles in `BitVectorData` and added `BitVectorDataMeta` with
  `Serializable` support for both `BitVectorData` and `BitVector`, enabling
  zero-copy reconstruction from arena metadata.
- Introduced a `Serializable` trait for metadata-based reconstruction and
  implemented it for `CompactVector`, `DacsByte`, and `WaveletMatrix`.
- Audited `DacsByte` and `WaveletMatrix` to leverage `SectionHandle::view`
  during deserialization, removing legacy `slice_to_bytes` helpers and fully
  adopting the `ByteArea`-backed reconstruction path.
- Switched internal bit-vector words and handles from `usize` to `u64`, removing
  unsafe handle transmutes in `WaveletMatrixBuilder` and fixing word size to
  64-bit.
  - Reversed remaining layers and popped in `WaveletMatrixBuilder::freeze`
    to avoid repeated vector shifts.
  - `WaveletMatrixMeta` now stores a handle slice of per-layer handles, and
    `WaveletMatrixBuilder` allocates that slice from the `SectionWriter`.
  - `WaveletMatrixBuilder::with_capacity` records each layer's handle up front,
    eliminating handle assignment during `freeze`.
  - Switched to the zerocopy `SectionHandle` from `anybytes`, removing the
    interim `HandleRepr` shim.
  - Added `WaveletMatrixBuilder` for fixed-size construction, writing raw bits per
    layer and stably partitioning them on `freeze`; `WaveletMatrix::from_iter`
    now builds via this builder without requiring iterator cloning.
  - `WaveletMatrix` construction now goes through `from_iter`, which allocates
    layer bitvectors from a `SectionWriter` and consumes a single
    `ExactSizeIterator` without temporary `CompactVector` partitions.
  - `CompactVector::iter` now implements `ExactSizeIterator` to support the new
    constructor.
  - `WaveletMatrixBuilder::freeze` partitions layers in place, removing the
    temporary bit buffer previously used during construction.
  - Removed `order` and `next_order` buffers by sorting remaining layers in
    place during each `freeze` step.
  - Optimized `WaveletMatrixBuilder::freeze` using stable per-layer partitions
    and cycle-based permutations, reducing layer processing to linear time.
  - Replaced the `perm` array with a scratch `visited` bitmap and cycle
    rotations so each level permutes lower layers in place with only `O(n)`
    extra bits.
  - Stored row suffix bits in a `usize` during cycle rotations, removing the
    temporary `Vec<bool>` from `rotate_cycle_over_lower_levels`.
  - Reused `BitVectorBuilder` as the scratch `visited` bitmap for
    wavelet-matrix construction, eliminating the separate `BitArrayBuilder`.
  - Added `swap_bits` helper to `BitVectorBuilder` for in-place bit exchanges.
  - Reworked `WaveletMatrix::from_iter` to require a cloneable iterator and
    build layers in two passes without temporary buffers.
- Rewrote `CompactVectorBuilder` to use fixed-size `set_int` and `set_ints`
  APIs, removing `push_int`/`extend` and updating builders and examples.
- Added `with_capacity` constructor on `BitVectorBuilder` and honored capacity in
  `CompactVectorBuilder::with_capacity` to pre-allocate bit storage.
- Replaced `BitVectorBuilder::new` with `with_capacity` that allocates from an
  `anybytes::ByteArea` section and plumbed `SectionWriter` through
  `CompactVectorBuilder` and wavelet matrix builders.
- Builders now track capacity and error when pushes exceed the reserved size.
- Made `DacsByte` generic over its flag index type with a default of `Rank9SelIndex`.
- `DacsByte::from_slice` now accepts a generic index type, removing `from_slice_with_index`.
- Added `BitVectorBuilder` and zero-copy `BitVectorData` backed by `anybytes::View`.
- Introduced `IndexBuilder` trait with a `Built` type and adjusted serialization helpers.
- Rename crate to `succdisk` to reflect on-disk succinct data structures.
- Rename crate from `succdisk` to `jerky`.
- Replaced the old `BitVector` with the generic `BitVector<I>` and renamed the
  mutable variant to `RawBitVector`.
- Replaced the push-based `BitVectorBuilder` with fixed-size `set_bit` and `set_bits` APIs and updated builders accordingly.
- Added `set_bits_from_iter` to `BitVectorBuilder` and later revised it to take a
  start offset and consume bits until the iterator ends or the builder is
  full, leaving any unconsumed items to the caller.
- Added `from_bit` constructor on `BitVectorBuilder` for repeating a single bit.
- `DacsByte` now stores level data as zero-copy `View<[u8]>` values.
- Replaced `to_bytes` helpers with `metadata` methods returning `SectionHandle`s
  so structures can be reconstructed zero-copy via `from_bytes`.
- Documented the byte layout for `DacsByte` sequences with ASCII art.
- Switched `anybytes` dependency to track the upstream Git repository for the
  latest changes.
- Removed internal byte buffers from data structures; `WaveletMatrix`,
  `DacsByte`, and `Rank9SelIndex` no longer store a `Bytes` field.
- Flags are serialized before level data to eliminate padding.
- Added `get_bits` methods to `BitVectorData` and `BitVector`.
- Removed deprecated `size_in_bytes` helpers.
- Added `scripts/devtest.sh` and `scripts/preflight.sh` for testing and
  verification workflows.
- Simplified CI workflow to run `scripts/preflight.sh` on pull requests.
- Fixed `scripts/preflight.sh` to install `rustfmt` when `cargo-fmt` is missing.
- `Rank9Sel` now stores a `BitVector<Rank9SelIndex>` built via `BitVectorBuilder`.
- Replaced `DArrayFullIndex` with new `DArrayIndex` that uses const generics
  to optionally include `select1` and `select0` support.
- Introduced `CompactVectorBuilder` mutable APIs `set_int` and `set_ints`.
- Simplified bit vector imports by re-exporting `BitVectorBuilder` and `Rank9SelIndex` and updating examples.
- Moved the `bit_vector::bit_vector` module contents directly into `bit_vector` for cleaner paths.
- Recorded future work items for a metadata serialization trait and
  ByteArea-backed documentation examples.
- Added README usage example demonstrating basic bit vector operations.
- Removed `bit_vector::prelude`; import traits directly with `use jerky::bit_vector::*`.
- Added `freeze()` on `CompactVectorBuilder` yielding an immutable `CompactVector` backed by `BitVector<NoIndex>`.
- Removed `CompactVector::new`; use `with_capacity` to construct builders.
- Wavelet matrix and DACs builders now use `BitVectorBuilder` for temporary bit
  vectors, storing only immutable `BitVector` data after construction.
- Removed obsolete `RawBitVector` type.
- Removed the `Rank9Sel` wrapper; use `BitVector<Rank9SelIndex>` directly.
- Removed the `DArray` wrapper; use `BitVector<darray::inner::DArrayIndex>` instead.
- Dropped the dense array implementation and all benchmarks referencing `DArrayIndex`.
- Removed the `Build` trait for bit vectors; construct indexes via `BitVectorBuilder` and `IndexBuilder`.
- Removed index builders in favor of parameterized index types constructed with `build`.
- Simplified benchmark code by importing index types and relying on type inference.
- Added `INVENTORY.md` for tracking pending work and ideas.
- Clarified inventory instructions in `AGENTS.md`.
- Removed the `## Completed Work` section from `INVENTORY.md`.
- Consolidated module layout by moving `WORD_LEN` into `bit_vectors::data` and
  adopting directory-based `mod.rs` files.
- Unified bit vector implementation under `bit_vector.rs` and removed the
  redundant `data` module.
- Renamed the `bit_vectors` module to `bit_vector` and updated imports.
- Updated benchmarks to use `jerky::bit_vector` imports.
- Removed the WaveletMatrix iterator caching TODO and inventory entry after
  benchmarking showed only a 3% performance gain.
- Fixed a stale doc link referencing the old `bit_vectors` module.
- Removed completed documentation cleanup tasks from `INVENTORY.md`.
- Fixed a typo in `bench/README.md`.
- Added iterators and `to_vec` helpers for inspecting built vectors.
- Split inspection tests so each assertion stands alone.
- Documented `WaveletMatrix` usage in `README.md`.
- Moved README usage examples to runnable files in `examples/`.
- Added `compact_vector` example showing construction and retrieval.
- Serialized `WaveletMatrix` and `DacsByte` directly into a `ByteArea` to
  avoid intermediate copies and guarantee contiguous layout.
- Enabled doctests for `WaveletMatrix` by removing `ignore` fences from its
  documentation examples.
- `DacsByte::from_slice` now writes level bytes and flags directly into
  `SectionWriter` buffers, removing the intermediate `Vec` allocations.
- Stored per-level `DacsByte` handles in the byte arena, allowing
  `DacsByteMeta` to reference a single handle slice like `WaveletMatrixMeta`.
- Expanded examples and README with `ByteArea`/`SectionHandle` metadata
  reconstruction for set-based APIs, adding a `dacs_byte` usage demo.
- Added a `Metadata` marker trait to ensure metadata structs implement the
  necessary zero-copy traits for safe byte serialization.
