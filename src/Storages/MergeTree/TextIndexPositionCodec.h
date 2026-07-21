#pragma once

#include <base/types.h>

namespace DB
{

/// Positions layout marker persisted in the text index header (one byte per part).
///
/// 'Blocked' (TextIndexBlockedPositionsCodec — candidate-driven per-document position lists) is the
/// only supported layout; it is not user-configurable. The values 0 and 1 identified pre-release
/// roaringish layouts; parts carrying them must be rebuilt by dropping and re-creating the index.
class TextIndexPositionCodec
{
public:
    enum class Encoding : UInt8
    {
        LegacyRaw = 0,
        LegacyPfor = 1,
        Blocked = 2,
    };
};

}
