# Integration

## Cross-Project Knowledge

| Document | Purpose |
|----------|---------|
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Critical traps from FIA, data leakage patterns |
| [CROSS_PRODUCT_COMPARISON.md](CROSS_PRODUCT_COMPARISON.md) | What transfers between MYGA/FIA/RILA projects |
| [HUB_PATTERN_REFERENCES.md](HUB_PATTERN_REFERENCES.md) | Hub-specific adaptations for RILA |

## Key Transfers from FIA

1. **Yield economics** - Cap rate coefficient > 0
2. **Lag structure** - No lag-0 competitors
3. **Leakage prevention** - Walk-forward CV
4. **Mathematical equivalence** - 1e-12 tolerance

## Five Critical Traps

| Trap | Description | Prevention |
|------|-------------|------------|
| Cap Rate Sign | Expecting negative coefficient | Enforce positive constraint |
| Lag-0 Competitors | Using current rates | Use t-2, t-3, t-4 only |
| Channel Breaks | Volume spikes from new channels | Control or filter |
| Simple Competitor Means | Using top-N mean | Use market-share weighted |
| Buffer Level Ignored | Treating all products same | Include as control/stratify |

See [LESSONS_LEARNED.md](LESSONS_LEARNED.md) for details.

## What Transfers vs What Doesn't

| Transfers | Requires Modification | Does NOT Transfer |
|-----------|----------------------|-------------------|
| Walk-forward CV | Competitor aggregation | MYGA firm-level analysis |
| Holiday masking | Feature names | MYGA threshold effects |
| Treasury controls | Buffer level controls | Specific coefficient values |

See [CROSS_PRODUCT_COMPARISON.md](CROSS_PRODUCT_COMPARISON.md) for comprehensive comparison.

## Related Projects

- `~/Claude/fia-price-elasticity/` - FIA price elasticity analysis
- `~/Claude/lever_of_archimedes/` - Hub for shared patterns
- `~/Claude/research-kb/` - Research knowledge base
