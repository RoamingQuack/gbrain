# pgvectorscale (StreamingDiskANN) for high-dimension ANN

> Status: **partial** — policy + capability probe + schema emission land in
> `src/core/vector-index.ts` / `PostgresEngine.initSchema`. Doctor GUCs and
> full operator UX continue in follow-up PRs. See plan notes in the graphiti
> workspace session.

## Why

pgvector **HNSW** caps at **2000** dims (`vector`) and **4000** (`halfvec`).
gbrain historically **skipped HNSW** above 2000 and used exact `<=>` scans.

That hurts:

- Voyage 2048, ZeroEntropy 2560 full width, OpenAI large 3072 without Matryoshka
- **User-hosted OpenAI-compatible embedders** (LiteLLM, vLLM, private gateways)
  that often expose native widths above 2000 via `trust_custom_dims`

**[pgvectorscale](https://github.com/timescale/pgvectorscale)** adds the
`diskann` access method so float32 ANN can index up to ~**16k** dims on
`vector`, still using pgvector types and distance operators.

## Hosting

| Host | vectorscale? |
|------|----------------|
| Self-hosted Postgres | Install / Docker build |
| Timescale / Tiger Cloud | Yes |
| DigitalOcean / PlanetScale PG | Check extension catalog |
| **Supabase** | **Not available** (today) |
| **PGLite** | **No** |

gbrain **feature-detects** `vectorscale` + `diskann`. Without them, behavior is unchanged.

```sql
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
```

## Policy (auto)

| dims | vectorscale | Index |
|------|-------------|--------|
| ≤ 2000 | any | **HNSW** (default) |
| > 2000 | present | **diskann** |
| > 2000 | absent | exact scan (legacy) |

Optional: prefer diskann even for small dims (memory-bound corpora) — config
hook `vector_index.prefer_diskann` (policy API ready; config surface TBD).

## Query

Same SQL as today:

```sql
ORDER BY embedding <=> $1::vector LIMIT k
```

Session GUCs when diskann is selected (applied by search path in follow-up):

- `diskann.query_search_list_size`
- `diskann.query_rescore`

## Code map

| Symbol | File |
|--------|------|
| `PGVECTOR_HNSW_*` / `selectChunkEmbeddingIndex` | `src/core/vector-index.ts` |
| `probeVectorCapabilities` | `src/core/vector-index.ts` |
| Schema emit | `getPostgresSchema(..., indexOpts)` in `postgres-engine.ts` |

## See also

- `docs/integrations/embedding-providers.md` — provider dims + HNSW note
- Upstream issues: gbrain #640 (skip HNSW >2000), #1046 (halfvec high-dim)
