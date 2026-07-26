/**
 * pgvector / optional pgvectorscale index policy + lifecycle manager.
 *
 * Original v0.27 surface: chunkEmbeddingIndexSql / applyChunkEmbeddingIndexPolicy
 * (schema-time index emission).
 *
 * v0.30.1 lifecycle additions:
 *   - dropAndRebuild (A3): atomic-swap pattern; build new index with temp
 *     name, ALTER...RENAME swap atomically, drop old. If rebuild fails the
 *     old index stays intact and search keeps working.
 *   - checkActiveBuild: pre-op probe of pg_stat_activity.
 *   - dropZombieIndexes: startup sweep of indisvalid=false indexes,
 *     guarded against in-progress builds.
 *   - monitorBuild: progress reporter during long-running CREATE INDEX.
 *
 * pgvectorscale (StreamingDiskANN / access method `diskann`):
 *   Raises float32 ANN beyond HNSW's 2000-dim cap (vector) / 4000 (halfvec).
 *   Complements pgvector — does not replace types. Optional; feature-detected.
 *   Important for high-dim + user-hosted OpenAI-compatible embedders
 *   (LiteLLM, vLLM, etc.) that often expose widths > 2000.
 *   Not available on PGLite or most Supabase tiers — always fall back.
 */

import type { BrainEngine } from './engine.ts';

export const PGVECTOR_HNSW_VECTOR_MAX_DIMS = 2000;
export const PGVECTOR_HNSW_HALFVEC_MAX_DIMS = 4000;
/** pgvectorscale diskann practical ceiling on `vector` (pgvector storage max is 16000). */
export const PGVECTORSCALE_DISKANN_VECTOR_MAX_DIMS = 16000;

/** Access method chosen for a chunk embedding ANN index. */
export type VectorIndexAccessMethod = 'hnsw' | 'diskann' | 'none';

/**
 * `auto` — HNSW when dims allow; diskann when dims > HNSW cap and vectorscale present; else exact.
 * `hnsw` — never use diskann (legacy / managed PG without vectorscale).
 * `diskann` — prefer diskann whenever vectorscale is present (even dims ≤ 2000).
 * `none` — never create ANN indexes (exact scan only).
 */
export type VectorIndexBackendPreference = 'auto' | 'hnsw' | 'diskann' | 'none';

export interface VectorCapabilities {
  /** Engine kind that was probed. */
  engineKind: 'postgres' | 'pglite';
  /** pgvector (`vector`) extension installed. */
  pgvector: boolean;
  pgvectorVersion: string | null;
  /** pgvectorscale (`vectorscale`) extension installed. */
  vectorscale: boolean;
  vectorscaleVersion: string | null;
  /** Access method `diskann` registered (usually implies vectorscale). */
  diskann: boolean;
}

export interface VectorIndexPolicyConfig {
  /** Default `auto`. */
  backend?: VectorIndexBackendPreference;
  /**
   * When backend is `auto` and dims ≤ 2000, still prefer diskann if available
   * (large corpora / memory-bound hosts). Default false.
   */
  preferDiskann?: boolean;
  /** diskann WITH options (optional). */
  diskann?: {
    storageLayout?: 'memory_optimized' | 'plain';
    numNeighbors?: number;
    searchListSize?: number;
    maxAlpha?: number;
    numBitsPerDimension?: number;
  };
}

export interface SelectedChunkEmbeddingIndex {
  am: VectorIndexAccessMethod;
  /** SQL fragment(s) for schema emission (CREATE INDEX or skip comment). */
  sql: string;
  /** Optional session GUCs to apply on search connections when am === 'diskann'. */
  sessionGucs: Record<string, string>;
  /** Human reason for selection (logs / doctor). */
  reason: string;
}

const CHUNK_EMBEDDING_HNSW_INDEX =
  'CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON content_chunks USING hnsw (embedding vector_cosine_ops);';

const CHUNK_EMBEDDING_DISKANN_INDEX_BASE =
  'CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON content_chunks USING diskann (embedding vector_cosine_ops)';

/** Capabilities when vectorscale cannot exist (PGLite) or probe not run. */
export function emptyVectorCapabilities(
  engineKind: 'postgres' | 'pglite' = 'pglite',
): VectorCapabilities {
  return {
    engineKind,
    pgvector: engineKind === 'pglite', // PGLite ships pgvector-in-WASM
    pgvectorVersion: null,
    vectorscale: false,
    vectorscaleVersion: null,
    diskann: false,
  };
}

/**
 * Probe Postgres for vector + vectorscale extensions and diskann AM.
 * PGLite short-circuits (no vectorscale). Failures return empty caps (best-effort).
 */
export async function probeVectorCapabilities(
  engine: BrainEngine,
): Promise<VectorCapabilities> {
  if (engine.kind !== 'postgres') {
    return emptyVectorCapabilities('pglite');
  }
  try {
    const extRows = await engine.executeRaw<{ extname: string; extversion: string }>(
      `SELECT extname, extversion FROM pg_extension
       WHERE extname IN ('vector', 'vectorscale')`,
    );
    let pgvector = false;
    let pgvectorVersion: string | null = null;
    let vectorscale = false;
    let vectorscaleVersion: string | null = null;
    for (const r of extRows) {
      if (r.extname === 'vector') {
        pgvector = true;
        pgvectorVersion = r.extversion ?? null;
      }
      if (r.extname === 'vectorscale') {
        vectorscale = true;
        vectorscaleVersion = r.extversion ?? null;
      }
    }
    let diskann = false;
    try {
      const amRows = await engine.executeRaw<{ amname: string }>(
        `SELECT amname FROM pg_am WHERE amname = 'diskann' LIMIT 1`,
      );
      diskann = amRows.length > 0;
    } catch {
      diskann = false;
    }
    // Extension without AM is incomplete; treat as no diskann.
    if (vectorscale && !diskann) {
      diskann = false;
    }
    return {
      engineKind: 'postgres',
      pgvector,
      pgvectorVersion,
      vectorscale,
      vectorscaleVersion,
      diskann: diskann && vectorscale,
    };
  } catch {
    return emptyVectorCapabilities('postgres');
  }
}

function buildDiskannIndexSql(config?: VectorIndexPolicyConfig): string {
  const d = config?.diskann ?? {};
  const withParts: string[] = [];
  const layout = d.storageLayout ?? 'memory_optimized';
  withParts.push(`storage_layout = '${layout}'`);
  if (d.numNeighbors != null) withParts.push(`num_neighbors = ${Number(d.numNeighbors)}`);
  if (d.searchListSize != null) withParts.push(`search_list_size = ${Number(d.searchListSize)}`);
  if (d.maxAlpha != null) withParts.push(`max_alpha = ${Number(d.maxAlpha)}`);
  if (d.numBitsPerDimension != null) {
    withParts.push(`num_bits_per_dimension = ${Number(d.numBitsPerDimension)}`);
  }
  return `${CHUNK_EMBEDDING_DISKANN_INDEX_BASE}\n  WITH (${withParts.join(', ')});`;
}

function diskannSessionGucs(): Record<string, string> {
  // Conservative defaults; operators can override via vector_index.diskann later.
  return {
    'diskann.query_search_list_size': '100',
    'diskann.query_rescore': '50',
  };
}

function exactScanSkipSql(dims: number, detail: string): string {
  return [
    '-- idx_chunks_embedding skipped: ' + detail,
    `-- dims=${dims}; exact vector scans (<=>) remain available.`,
  ].join('\n');
}

/**
 * Select ANN index strategy for primary content_chunks.embedding (`vector` type).
 *
 * Primary column is always `vector(N)` in gbrain; halfvec is used on facts/query_cache
 * via migrations and is handled by hnswMaxDimsForType separately.
 */
export function selectChunkEmbeddingIndex(
  dims: number,
  capabilities: VectorCapabilities = emptyVectorCapabilities(),
  config: VectorIndexPolicyConfig = {},
): SelectedChunkEmbeddingIndex {
  const backend = config.backend ?? 'auto';
  const preferDiskann = config.preferDiskann === true;
  const hasDiskann = capabilities.diskann === true;

  if (backend === 'none') {
    return {
      am: 'none',
      sql: exactScanSkipSql(dims, 'vector_index.backend=none'),
      sessionGucs: {},
      reason: 'backend=none',
    };
  }

  if (!Number.isFinite(dims) || dims <= 0) {
    return {
      am: 'none',
      sql: exactScanSkipSql(dims, 'invalid embedding dimensions'),
      sessionGucs: {},
      reason: 'invalid_dims',
    };
  }

  const canDiskann = hasDiskann && dims <= PGVECTORSCALE_DISKANN_VECTOR_MAX_DIMS;

  // Prefer / force diskann when available
  if (canDiskann) {
    if (backend === 'diskann' || (backend === 'auto' && preferDiskann)) {
      return {
        am: 'diskann',
        sql: buildDiskannIndexSql(config),
        sessionGucs: diskannSessionGucs(),
        reason: backend === 'diskann' ? 'backend=diskann' : 'prefer_diskann',
      };
    }
    // auto: use diskann only when dims exceed HNSW float32 cap
    if (backend === 'auto' && dims > PGVECTOR_HNSW_VECTOR_MAX_DIMS) {
      return {
        am: 'diskann',
        sql: buildDiskannIndexSql(config),
        sessionGucs: diskannSessionGucs(),
        reason: `diskann dims=${dims}>${PGVECTOR_HNSW_VECTOR_MAX_DIMS} vectorscale`,
      };
    }
  }

  // HNSW path (default for dims ≤ 2000; also fallback when diskann forced but missing)
  if (dims <= PGVECTOR_HNSW_VECTOR_MAX_DIMS && backend !== 'none') {
    if (backend === 'hnsw' || backend === 'auto' || (backend === 'diskann' && !hasDiskann)) {
      return {
        am: 'hnsw',
        sql: CHUNK_EMBEDDING_HNSW_INDEX,
        sessionGucs: {},
        reason: backend === 'diskann' && !hasDiskann
          ? 'hnsw_fallback_no_vectorscale'
          : `hnsw dims=${dims}≤${PGVECTOR_HNSW_VECTOR_MAX_DIMS}`,
      };
    }
  }

  // Exact scan fallback (v0.28+ high-dim without vectorscale; or dims above diskann max)
  const hint = dims > PGVECTORSCALE_DISKANN_VECTOR_MAX_DIMS
    ? `dims exceed diskann max (${PGVECTORSCALE_DISKANN_VECTOR_MAX_DIMS})`
    : `pgvector HNSW vector indexes support at most ${PGVECTOR_HNSW_VECTOR_MAX_DIMS} dimensions`
      + (capabilities.engineKind === 'postgres' && !hasDiskann
        ? '; install CREATE EXTENSION vectorscale for diskann ANN (LiteLLM/vLLM high-dim local embeds)'
        : '');
  return {
    am: 'none',
    sql: exactScanSkipSql(dims, hint),
    sessionGucs: {},
    reason: hasDiskann ? 'dims_out_of_range' : 'high_dim_exact_scan',
  };
}

/**
 * Schema-time SQL for idx_chunks_embedding.
 * @param dims embedding width
 * @param opts optional capabilities (vectorscale) + policy; defaults preserve
 *   pre-vectorscale behavior (HNSW ≤2000, else exact-scan skip comment).
 */
export function chunkEmbeddingIndexSql(
  dims: number,
  opts?: { capabilities?: VectorCapabilities; config?: VectorIndexPolicyConfig },
): string {
  return selectChunkEmbeddingIndex(
    dims,
    opts?.capabilities ?? emptyVectorCapabilities(),
    opts?.config ?? {},
  ).sql;
}

export function hnswMaxDimsForType(columnType: 'vector' | 'halfvec'): number {
  return columnType === 'halfvec' ? PGVECTOR_HNSW_HALFVEC_MAX_DIMS : PGVECTOR_HNSW_VECTOR_MAX_DIMS;
}

export function applyChunkEmbeddingIndexPolicy(
  sql: string,
  dims: number,
  opts?: { capabilities?: VectorCapabilities; config?: VectorIndexPolicyConfig },
): string {
  return sql.replaceAll(CHUNK_EMBEDDING_HNSW_INDEX, chunkEmbeddingIndexSql(dims, opts));
}

// ---------------------------------------------------------------------------
// v0.30.1 Lifecycle Manager (Fix 5)
// ---------------------------------------------------------------------------

export interface IndexSpec {
  /** The CURRENT (production) index name. */
  name: string;
  table: string;
  column: string;
  /** USING clause body — e.g. `hnsw (embedding vector_cosine_ops)`. */
  using: string;
  /** Optional WHERE predicate (without WHERE keyword). */
  condition?: string;
}

export interface ActiveBuildInfo {
  active: boolean;
  pid?: number;
  query?: string;
  application_name?: string;
}

/**
 * Probe pg_stat_activity for an active CREATE INDEX on this index name.
 * Used as a pre-op guard so dropAndRebuild doesn't compete with a build
 * already in flight (Supabase auto-maintenance + parallel gbrain procs).
 */
export async function checkActiveBuild(
  engine: BrainEngine,
  indexName: string,
): Promise<ActiveBuildInfo> {
  if (engine.kind !== 'postgres') return { active: false };
  try {
    const rows = await engine.executeRaw<{ pid: number; query: string; application_name: string | null }>(
      `SELECT pid, query, application_name
       FROM pg_stat_activity
       WHERE state = 'active'
         AND (query ILIKE $1 OR query ILIKE $2)
         AND pid != pg_backend_pid()
       LIMIT 1`,
      [`%CREATE INDEX%${indexName}%`, `%REINDEX%${indexName}%`],
    );
    if (rows.length === 0) return { active: false };
    const r = rows[0];
    return {
      active: true,
      pid: r.pid,
      query: r.query,
      application_name: r.application_name ?? undefined,
    };
  } catch {
    return { active: false };
  }
}

/**
 * Sweep invalid HNSW indexes on startup. Drops any pg_index row with
 * indisvalid=false on tables we care about, AS LONG AS no active build
 * is running for that index (codex Fix-5 zombie-cleanup guard).
 *
 * Postgres-only. PGLite returns { dropped: [] }.
 */
export async function dropZombieIndexes(
  engine: BrainEngine,
  tableNames: string[] = ['content_chunks', 'pages', 'takes'],
): Promise<{ dropped: string[] }> {
  if (engine.kind !== 'postgres') return { dropped: [] };
  const dropped: string[] = [];
  try {
    // Find invalid indexes on our tables.
    const rows = await engine.executeRaw<{ indexname: string; tablename: string }>(
      `SELECT i.relname AS indexname, t.relname AS tablename
       FROM pg_index ix
       JOIN pg_class i ON i.oid = ix.indexrelid
       JOIN pg_class t ON t.oid = ix.indrelid
       WHERE ix.indisvalid = false
         AND t.relname = ANY($1)`,
      [tableNames],
    );
    for (const r of rows) {
      // Guard: skip if there's an active build for this index.
      const active = await checkActiveBuild(engine, r.indexname);
      if (active.active) {
        process.stderr.write(`[hnsw] skipping zombie cleanup of ${r.indexname} — active build (pid ${active.pid})\n`);
        continue;
      }
      try {
        await engine.executeRaw(`DROP INDEX IF EXISTS ${r.indexname}`);
        dropped.push(r.indexname);
        process.stderr.write(`[hnsw] dropped zombie index ${r.indexname} on ${r.tablename}\n`);
      } catch (err) {
        process.stderr.write(`[hnsw] failed to drop ${r.indexname}: ${(err as Error).message}\n`);
      }
    }
  } catch (err) {
    // Best-effort: pg_stat_activity / pg_index queries may be restricted
    // on managed Postgres tiers. Don't fail engine.connect() over it.
    process.stderr.write(`[hnsw] zombie-index probe failed: ${(err as Error).message}\n`);
  }
  return { dropped };
}

/**
 * Atomic-swap rebuild (A3): build new index with temp name, swap atomically.
 *
 *   1. Probe pg_stat_activity → bail if another build is active
 *   2. Compose temp name: <name>_rebuild_<unix-ms>
 *   3. CREATE INDEX <temp> with the spec's USING clause + condition
 *   4. In a single transaction:
 *        DROP INDEX <name>
 *        ALTER INDEX <temp> RENAME TO <name>
 *   5. If step 3 fails (OOM, timeout, conn drop), the old index is intact
 *      and search keeps serving queries. Caller can retry.
 *
 * The CREATE INDEX uses CONCURRENTLY so it doesn't block writes during the
 * build; this requires `transaction:false` semantics so we route through
 * engine.withReservedConnection.
 */
export async function dropAndRebuild(
  engine: BrainEngine,
  spec: IndexSpec,
  opts: { reason: string; force?: boolean } = { reason: 'manual' },
): Promise<{ rebuilt: boolean; tempName: string }> {
  if (engine.kind !== 'postgres') {
    return { rebuilt: false, tempName: spec.name };
  }

  const active = await checkActiveBuild(engine, spec.name);
  if (active.active && !opts.force) {
    process.stderr.write(
      `[hnsw] dropAndRebuild ${spec.name} aborted: active build pid ${active.pid} (${active.application_name ?? 'unknown'}). Pass --force to proceed anyway.\n`,
    );
    return { rebuilt: false, tempName: spec.name };
  }

  const ts = Date.now();
  const tempName = `${spec.name}_rebuild_${ts}`;
  const where = spec.condition ? ` WHERE ${spec.condition}` : '';

  process.stderr.write(`[hnsw] rebuild ${spec.name} → ${tempName} (reason=${opts.reason})\n`);

  // Step 3: build the new index (CONCURRENTLY) under a reserved connection.
  await engine.withReservedConnection(async conn => {
    await conn.executeRaw(
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS ${tempName} ON ${spec.table} USING ${spec.using}${where}`,
    );
  });

  // Step 4: atomic swap inside a transaction.
  await engine.transaction(async (tx) => {
    const innerSql = (tx as unknown as { sql: any }).sql;
    if (innerSql) {
      await innerSql.unsafe(`DROP INDEX IF EXISTS ${spec.name}`);
      await innerSql.unsafe(`ALTER INDEX ${tempName} RENAME TO ${spec.name}`);
    }
  });

  process.stderr.write(`[hnsw] rebuild complete: ${spec.name}\n`);
  return { rebuilt: true, tempName };
}

/**
 * Poll pg_stat_activity to monitor a CREATE INDEX in progress. Reports
 * elapsed time + progress (rows-built proxy via pg_stat_progress_create_index
 * when available; falls back to relation size growth otherwise).
 *
 * Caller wraps a CREATE INDEX in a separate code path; this function is
 * orthogonal — it just polls and emits progress lines.
 */
export interface BuildProgress {
  elapsed_ms: number;
  size_bytes?: number;
  workers?: number;
  pid?: number;
}

export async function monitorBuild(
  engine: BrainEngine,
  indexName: string,
  onProgress: (status: BuildProgress) => void,
  opts: { intervalMs?: number; maxIterations?: number } = {},
): Promise<void> {
  if (engine.kind !== 'postgres') return;
  const interval = opts.intervalMs ?? 30000;
  const maxIterations = opts.maxIterations ?? 240; // 240 * 30s = 2h cap
  const t0 = Date.now();
  for (let i = 0; i < maxIterations; i++) {
    const active = await checkActiveBuild(engine, indexName);
    if (!active.active) return;
    let size_bytes: number | undefined;
    try {
      const rows = await engine.executeRaw<{ size: number }>(
        `SELECT pg_relation_size(c.oid) AS size FROM pg_class c WHERE c.relname = $1 LIMIT 1`,
        [indexName],
      );
      if (rows[0]) size_bytes = Number(rows[0].size);
    } catch { /* size probe optional */ }
    onProgress({ elapsed_ms: Date.now() - t0, size_bytes, pid: active.pid });
    await new Promise(r => setTimeout(r, interval));
  }
}

/**
 * Detect whether a CREATE INDEX query in pg_stat_activity is from Supabase
 * auto-maintenance (vs. our gbrain process). Used by dropAndRebuild to
 * back off when auto-maintenance is doing the rebuild for us.
 */
export function isSupabaseAutoMaintenance(active: ActiveBuildInfo): boolean {
  if (!active.active) return false;
  const appName = (active.application_name ?? '').toLowerCase();
  return appName.includes('supabase') || appName.includes('postgres-meta');
}
