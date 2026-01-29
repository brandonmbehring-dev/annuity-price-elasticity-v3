# Research-KB Query Skill

Query the research-kb knowledge base for domain expertise relevant to RILA price elasticity.

## Usage

Invoke with: `/research-kb-query <topic>`

## Available Queries

### Demand Estimation

```bash
research-kb query "discrete choice demand estimation"
research-kb query "random coefficients logit"
research-kb query "pyblp best practices"
research-kb query "BLP estimation"
```

### Insurance Economics

```bash
research-kb query "equity indexed annuity"          # Boyle (2008) on EIA pricing
research-kb query "life insurer financial frictions"  # Koijen & Yogo (2015)
research-kb query "annuity adverse selection"       # Finkelstein & Poterba (2004)
research-kb query "RILA product structure"
```

### Causal Inference

```bash
research-kb query "instrumental variables"
research-kb query "double machine learning"
research-kb query "time series causal inference"
```

### Price Elasticity

```bash
research-kb query "price elasticity estimation"
research-kb query "competitive pricing models"
research-kb query "yield economics"
```

## Regulatory Context

| Regulation | Description |
|------------|-------------|
| FINRA Notice 22-08 | Complex products/RILA guidance |
| SEC Investor Bulletin | RILA investor alerts |
| NAIC Model Laws | Standard nonforfeiture (#808/#805) |

## Market Data References

| Source | Description |
|--------|-------------|
| LIMRA | Quarterly annuity sales releases |
| WINK/AnnuitySpecs | Product rate data dictionary |
| FRED | Economic indicators |

## Related Knowledge

- `knowledge/domain/RILA_ECONOMICS.md` - RILA product economics
- `knowledge/integration/FIA_TRANSFER_NOTES.md` - Cross-project patterns
- `~/Claude/research-kb/codex-bib.md` - Full bibliography
