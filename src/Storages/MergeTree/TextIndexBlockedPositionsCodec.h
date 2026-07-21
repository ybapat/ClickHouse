#pragma once

#include <Storages/MergeTree/TextIndexPositionData.h>
#include <Common/PODArray.h>
#include <IO/ReadBuffer.h>
#include <IO/WriteBuffer.h>

#include <algorithm>
#include <span>
#include <vector>

namespace DB
{

/// Blocked candidate-driven positions codec (the text index positions layout).
///
/// Stores one token's positions as plain per-document position lists, chunked into blocks of
/// BLOCK_DOCS consecutive *posting ranks* — the stream carries no document ids and no frequency
/// lane: block b covers the token's postings entries [b*BLOCK_DOCS, (b+1)*BLOCK_DOCS), so a posting
/// cursor's rank addresses a document's slot directly. A phrase query therefore reads postings
/// first, intersects them into candidate rows, and fetches only the blocks covering those
/// candidates; bytes read scale with candidates, not occurrences.
///
/// Per-token stream layout (at token_info.position_offset in the .pos substream):
///     [VarUInt: num_docs]
///     [VarUInt: num_blocks]                        (must equal ceil(num_docs / BLOCK_DOCS))
///     num_blocks x [VarUInt: payload_bytes]        (directory; one contiguous read, offsets by prefix sum)
///     num_blocks x payload, back to back
///
/// Block payload (docs_in_block = min(BLOCK_DOCS, num_docs - b*BLOCK_DOCS)):
///     [VarUInt: num_exceptions]
///     num_exceptions x [VarUInt: local_rank][VarUInt: freq]   (ascending local_rank; freq >= 2;
///                                                              a document's freq defaults to 1)
///     [PFor: total_positions UInt32 values]        (total_positions = docs_in_block + sum(freq - 1);
///                                                   within-document deltas in posting order, the
///                                                   first position of each document absolute)
///
/// All counts are validated fail-closed (CORRUPTED_DATA) before any allocation.
class TextIndexBlockedPositionsCodec
{
public:
    /// Posting ranks per block. Fixed: the value is baked into written parts implicitly (a reader
    /// computes block membership as rank / BLOCK_DOCS), so changing it requires a format version bump.
    static constexpr size_t BLOCK_DOCS = 128;

    /// Parsed per-token directory. Offsets are absolute in the .pos stream, resolved against the
    /// token's position_offset, so the caller can seek straight to a block.
    struct Directory
    {
        UInt64 num_docs = 0;
        /// block_offsets[b] .. block_offsets[b + 1] is block b's payload; size numBlocks() + 1.
        std::vector<UInt64> block_offsets;

        size_t numBlocks() const { return block_offsets.empty() ? 0 : block_offsets.size() - 1; }
        size_t docsInBlock(size_t block_idx) const
        {
            const UInt64 begin = block_idx * BLOCK_DOCS;
            return static_cast<size_t>(std::min<UInt64>(BLOCK_DOCS, num_docs - begin));
        }
    };

    /// Reusable scratch for block decodes, owned by the caller (no per-call heap allocation on the
    /// hot path; PaddedPODArray gives the PFor kernels safe trailing padding).
    struct DecodeScratch
    {
        PaddedPODArray<char> payload;
        PaddedPODArray<UInt32> freqs;
        PaddedPODArray<UInt32> values;
    };

    /// Encodes a sorted RoaringishEntry array (the writer's in-memory accumulation form) as the
    /// blocked per-document stream.
    static void encode(std::span<const RoaringishEntry> entries, WriteBuffer & out);

    /// Number of distinct documents in a sorted RoaringishEntry array — the value the writer
    /// records as position_cardinality for blocked parts (readDirectory validates against it).
    static UInt64 countDocuments(std::span<const RoaringishEntry> entries);

    /// Reads the per-token directory; the stream must be positioned at the token's
    /// position_offset (= `blob_offset`, used to resolve absolute block offsets).
    /// `expected_num_docs` is the value recorded in the index header (position_cardinality);
    /// `available_bytes` (file size minus blob_offset) caps every declared size.
    static Directory readDirectory(ReadBuffer & in, UInt64 blob_offset, UInt64 expected_num_docs, size_t available_bytes);

    /// Decodes block `block_idx` (the stream must be positioned at dir.block_offsets[block_idx])
    /// and appends the positions of the requested local ranks (ascending, < docsInBlock) to
    /// `positions`, pushing one end offset per rank to `offsets`.
    static void decodeBlock(
        ReadBuffer & in,
        const Directory & dir,
        size_t block_idx,
        std::span<const UInt32> local_ranks,
        PaddedPODArray<UInt32> & offsets,
        PaddedPODArray<UInt32> & positions,
        DecodeScratch & scratch);

    /// Whole-token sequential decode (the merge path): positions of every posting rank.
    /// The stream must be positioned at the token's position_offset. On return
    /// doc_offsets.size() == num_docs + 1 and doc_offsets[r] .. doc_offsets[r + 1] indexes
    /// rank r's positions.
    static void decodeAll(
        ReadBuffer & in,
        UInt64 expected_num_docs,
        size_t available_bytes,
        PaddedPODArray<UInt32> & doc_offsets,
        PaddedPODArray<UInt32> & positions,
        DecodeScratch & scratch);
};

}
