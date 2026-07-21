#pragma once

#include <Common/PODArray.h>

#include <vector>

namespace DB
{

/// Candidate-driven phrase matching over blocked positions.
struct TextIndexPhraseSearch
{
    /// `candidates` are the ascending row ids containing every phrase term (the postings
    /// intersection). For each unique phrase token u, `per_token_positions[u]` holds the
    /// candidates' position lists concatenated in candidate order, delimited by
    /// `per_token_offsets[u]`: candidate i's positions are
    /// [i == 0 ? 0 : offsets[i - 1], offsets[i]) — the layout TextIndexBlockedPositionsCodec's
    /// block decode emits. `term_to_unique` maps each phrase term (in phrase order) to its
    /// unique-token index, so repeated terms reuse one decoded stream.
    ///
    /// Returns the candidates where some position p starts the phrase: term k at p + k for all k.
    static PaddedPODArray<UInt32> matchCandidatePositions(
        const PaddedPODArray<UInt32> & candidates,
        const std::vector<PaddedPODArray<UInt32>> & per_token_offsets,
        const std::vector<PaddedPODArray<UInt32>> & per_token_positions,
        const std::vector<size_t> & term_to_unique);
};

}
