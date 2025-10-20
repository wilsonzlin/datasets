# search.wilsonl.in Web Search Index Crawl + Text Embeddings

A production web search engine dataset built from scratch, containing 280 million indexed pages with 3 billion neural embeddings. Index cutoff: August 2023.

**Live Demo:** https://search.wilsonl.in

**Original Project:** https://blog.wilsonl.in/search-engine

## Dataset Overview

This dataset represents a complete production-scale search engine built entirely from scratch, featuring:

- **280 million web pages** crawled and parsed
- **3 billion embeddings** (768-dimensional vectors using sentence-transformers/multi-qa-mpnet-base-dot-v1)
- **Dual-level indexing** with both page-level (block-mean) and sentence-level (statement) embeddings
- **Sharded architecture** across 64 shards for scalability
- **Knowledge graph data** from DBpedia and Wikidata

The dataset includes the raw crawled content, normalized HTML, extracted statements (sentence chunks), embeddings, and multiple index formats for different use cases.

## Getting Started

**Quick start notebook:** [notebooks/get-to-know-a-dataset.ipynb](notebooks/get-to-know-a-dataset.ipynb)

**Bucket:** `s3://aws-opendata.wilsonl.in/search-engine` (us-east-1)

**Mounting with mount-s3:**
```bash
mount-s3 aws-opendata.wilsonl.in /mnt/search-engine --region us-east-1 --read-only
```

## Directory Structure

```
s3://aws-opendata.wilsonl.in/search-engine/

├── rocksdb-shards/              # 64 sharded RocksDB instances with page data
│   ├── shard-0/
│   ├── shard-1/
│   └── ... shard-63/
│
├── resource-id/                 # Resource ID mapping service
│   └── rocksdb/
│
├── hnsw-shards/                 # 64 sharded HNSW indices (original production format)
│   ├── block-mean/
│   │   ├── shard-0/index.hnsw   (~15 GB each, 960 GB total)
│   │   └── ... shard-63/
│   └── statement/
│       ├── shard-0/index.hnsw   (~10 GB each, 633 GB total)
│       └── ... shard-63/
│
├── hnsw-combined/               # Combined HNSW indices (built for export)
│   ├── block-mean.hnsw          (1.1 TB)
│   └── statement.hnsw           (767 GB)
│
├── kg-dbpedia/                  # DBpedia knowledge graph data
│   ├── vectors.hnsw
│   ├── vectors.sqlite3
│   └── article-abstracts.sqlite3
│
├── kg-wikidata/                 # Wikidata entity properties
│   └── rocksdb/
│
└── export/                      # Alternative data formats
    ├── data.parquet             # Columnar format for analytics
    ├── data-postcard/           # Binary format for fast lookups
    │   ├── uids.bin
    │   ├── offsets.bin
    │   └── data.bin
    ├── statement_uid_base_to_resource_uid.arrow
    ├── statement_embeddings_msgpack.bin
    ├── block_embeddings_msgpack.bin
    ├── norm_doc_json_brotli.bin
    ├── statements_json_brotli.bin
    ├── source_brotli.bin
    ├── statement_labels_msgpack.bin
    ├── urls.txt
    └── data.sql
```

## Data Organization

### RocksDB Shards

The primary data store, sharded across 64 instances using XXH3 consistent hashing.

**Sharding:** `shard_number = xxh3_64(key) % 64`

**Key Format:** `[prefix_byte][url]`

Each shard contains 9 key types:

| Prefix | Key Type | Value Format | Description |
|--------|----------|--------------|-------------|
| `0x01` | `Resource` | MessagePack | Basic metadata (title, state, HTTP status, fetch timestamps) |
| `0x02` | `ResourceLinks` | MessagePack | Outbound links from the page |
| `0x03` | `ResourceMeta` | MessagePack | OpenGraph and meta tags (e.g., og:description, og:image) |
| `0x04` | `ResourceNormDoc` | JSON + Brotli | Normalized HTML structure |
| `0x05` | `ResourceSource` | Brotli | Original HTML source |
| `0x06` | `ResourceStatements` | JSON + Brotli | Sentence chunks with context windows |
| `0x07` | `ResourceBlockEmbeddings` | MessagePack | Block-level 768-dim embeddings |
| `0x08` | `ResourceStatementLabels` | MessagePack | Classification labels for statements |
| `0x09` | `ResourceStatementEmbeddings` | MessagePack | Statement-level 768-dim embeddings |

**Resource Schema (prefix 0x01):**
```rust
{
  state: "0" | "2" | "3" | "4" | "5" | "6" | "7" | "8",  // Fetching, Parsing, Labelling, Redirected, BadStatus, FetchError, ParseError, DecompressError
  http_status: u16?,
  original_content_encoding: string?,
  redirect_location: string?,
  error: string?,                    // Error code if state is error
  error_details: string?,            // Stack trace or details
  unknown_html_elements: [string]?,  // HTML elements not recognized
  last_fetched: DateTime?,
  last_fetch_id: u64?,
  title: string?,
  icon_url: string?
}
```

**ResourceStatements Schema (prefix 0x06):**
```rust
{
  heading_entries: [{
    level: u8,        // Heading level (1-6)
    text: string
  }],
  statements: [{
    text: string,     // The sentence/chunk text
    path: [u32],      // DOM path to element
    header: [u32],    // Indices into heading_entries
    window: [u32],    // Indices into statements (context)
    typ: "Text" | "TableRow" | "OrderedList" | "CodeBlock" | "Blockquote"
  }]
}
```

**ResourceBlockEmbeddings Schema (prefix 0x07):**
```rust
{
  blocks: [{
    statement_start_index: u32,  // Index into statements array
    embedding: [u8]              // 768 float32 values (3072 bytes)
  }]
}
```

### Resource ID Service

Bidirectional mappings between URLs and numeric IDs, stored in `resource-id/rocksdb/`.

**Key Types:**

| Prefix | Mapping | Key Format | Value Format |
|--------|---------|------------|--------------|
| `0x01` | URL → Resource | `[0x01][url]` | MessagePack: `{uid, fetch_id, statement_uid_base}` |
| `0x02` | UID → URL | `[0x02][uid_big_endian]` | String: URL |
| `0x03` | Statement UID Base → URL | `[0x03][uid_big_endian]` | String: URL |

**Note:** UIDs are stored as big-endian 64-bit integers to maintain sort order for range queries.

The statement UID base is the starting UID for all statements within a resource. Statement UIDs are sequential, so `statement_uid_base + statement_index = statement_uid`.

### HNSW Indices

Hierarchical Navigable Small World (HNSW) graphs for approximate nearest neighbor search using inner product (dot product) similarity.

**Types:**

- **block-mean**: Page-level embeddings (average of block embeddings)
- **statement**: Sentence-level embeddings

**Formats:**

- **Sharded** (64 shards): `hnsw-shards/{type}/shard-{0..63}/index.hnsw`
  - Sharded using consistent hashing for parallel querying
  - block-mean: ~15 GB per shard, 960 GB total
  - statement: ~10 GB per shard, 633 GB total
  
- **Combined** (single file): `hnsw-combined/{type}.hnsw`
  - Created during export for convenience
  - block-mean: 1.1 TB
  - statement: 767 GB

**Index IDs:**
- block-mean index: IDs are resource UIDs
- statement index: IDs are statement UIDs (must be mapped back to resource UIDs via resource-id service)

**Vector format:** 768-dimensional float32 arrays, normalized for dot product similarity.

### Parquet Export

Columnar format for analytics and bulk processing: `export/data.parquet`

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `url` | string | Normalized URL |
| `uid` | uint64 | Resource unique ID |
| `fetch_id` | uint64 | Fetch operation ID |
| `statement_uid_base` | uint64? | Starting UID for statements |
| `state` | string | Resource state (see RocksDB schema) |
| `http_status` | uint16? | HTTP status code |
| `original_content_encoding` | string? | Content-Encoding header |
| `redirect_location` | string? | Location header for redirects |
| `error` | string? | Error code if failed |
| `error_details` | string? | Error details |
| `unknown_html_elements` | list[string]? | Unrecognized HTML elements |
| `last_fetched` | timestamp? | Last fetch timestamp |
| `last_fetch_id` | uint64? | Last fetch operation ID |
| `title` | string? | Page title |
| `icon_url` | string? | Favicon URL |
| `url_hostname` | string | Hostname from URL |
| `url_hostname_rev` | string | Reversed hostname (for TLD grouping) |
| `url_path` | string | Path component |
| `url_path_ext` | string | File extension (lowercase) |
| `links` | list[string]? | Outbound links |
| `meta` | map[string, string]? | Meta tags |
| `{field}_offset` | uint64? | Offset into external blob file |
| `{field}_size` | uint64 | Size in external blob file |

**External blob files** (referenced by offset/size columns):
- `norm_doc_json_brotli.bin` - Normalized HTML documents (Brotli compressed JSON)
- `source_brotli.bin` - Original HTML source (Brotli compressed)
- `statements_json_brotli.bin` - Statement arrays (Brotli compressed JSON)
- `block_embeddings_msgpack.bin` - Block embeddings (MessagePack)
- `statement_embeddings_msgpack.bin` - Statement embeddings (MessagePack)
- `statement_labels_msgpack.bin` - Classification labels (MessagePack)

### Postcard Binary Format

Efficient binary serialization for Rust applications: `export/data-postcard/`

**Files:**
- `uids.bin` - Array of uint64 resource UIDs
- `offsets.bin` - Array of uint64 offsets into data.bin
- `data.bin` - Concatenated Postcard-serialized records

**Access pattern:**
1. Load `uids.bin` and `offsets.bin` into memory
2. Build UID → row index mapping
3. For lookup: `row = uid_to_row[uid]`, `offset = offsets[row]`, `next_offset = offsets[row+1]`
4. Read `data.bin[offset:next_offset]` and deserialize with Postcard

**Record schema:** Same fields as Parquet, with offsets pointing to external blob files.

### Knowledge Graph Data

**DBpedia** (`kg-dbpedia/`):
- `vectors.hnsw` - HNSW index of Wikipedia article embeddings (768-dim, inner product)
- `vectors.sqlite3` - Article metadata (titles, IDs, Wikidata links)
- `article-abstracts.sqlite3` - Article summaries and descriptions

**Wikidata** (`kg-wikidata/rocksdb/`):
- RocksDB key-value store
- Keys: Wikidata entity IDs (e.g., "Q42")
- Values: MessagePack-encoded property dictionaries
- Properties include: birth/death dates, locations, occupations, relationships

## Data Dictionary

### URL Normalization

All URLs in the dataset are normalized:
- Scheme: http or https only
- Hostname: lowercase, IDN in punycode
- Port: removed if default (80 for http, 443 for https)
- Path: percent-encoding minimized (only `/`, `#`, `?` encoded)
- Query string: removed
- Fragment: removed
- Trailing slash: preserved (significant)

### Embedding Model

**Model:** `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- **Dimensions:** 768
- **Similarity metric:** Dot product (inner product)
- **Normalization:** All vectors are L2-normalized
- **Format:** float32 (4 bytes per dimension, 3072 bytes per vector)

### Statement Types

| Type | Description |
|------|-------------|
| `Text` | Regular paragraph text |
| `TableRow` | Content from table rows |
| `OrderedList` | Items from ordered lists |
| `CodeBlock` | Code snippets |
| `Blockquote` | Quoted content |

### Resource States

| State | Code | Description |
|-------|------|-------------|
| Fetching | "0" | Currently being fetched |
| Parsing | "2" | Being parsed |
| Labelling | "3" | Generating labels/embeddings |
| Redirected | "4" | HTTP 3xx redirect (see redirect_location) |
| BadStatus | "5" | Non-2xx/3xx HTTP status |
| FetchError | "6" | Network or fetch error |
| ParseError | "7" | HTML parsing failed |
| DecompressError | "8" | Content decompression failed |

## Data Access Patterns

### Pattern 1: Semantic Search (Production)

1. Load all 64 sharded HNSW indices (~1.6 TB RAM)
2. Embed query with sentence-transformers model
3. Query all shards in parallel (ThreadPoolExecutor)
4. Merge results and take top-k
5. Map UIDs to URLs via resource-id service
6. Fetch page data from RocksDB shards

### Pattern 2: Semantic Search (Simple)

1. Load combined HNSW indices (~1.9 TB RAM)
2. Embed query
3. Query indices directly (no parallel execution needed)
4. Map UIDs to URLs
5. Fetch page data from RocksDB shards

### Pattern 3: Analytics

1. Query Parquet with DuckDB/Polars/Spark
2. Use columnar operations for filtering, aggregation
3. Read external blobs only when needed (by offset/size)

### Pattern 4: Bulk Processing

1. Load Postcard mappings (uids.bin, offsets.bin)
2. Stream through data.bin
3. Deserialize with Postcard format
4. Process records sequentially or in parallel

## Technical Specifications

**Crawl period:** ~2 months (August 2023 cutoff)

**Embedding generation:**
- 200 GPUs at peak
- 100K embeddings/second throughput
- 90% average GPU utilization

**Index size:**
- RocksDB: 64 shards, ~1.3 TB per shard
- HNSW: 1.6 TB (sharded), 1.9 TB (combined)
- Parquet: ~500 GB (with external blobs)

**Vector count:**
- Block embeddings: ~280M vectors
- Statement embeddings: ~2.7B vectors
- Total: ~3B vectors

If you use this dataset in your research, please cite:

```bibtex
@misc{lin2025searchenginedataset,
  author = {Wilson Lin},
  title = {search.wilsonl.in Web Search Index Crawl + Text Embeddings},
  year = {2025},
  url = {https://github.com/wilsonzlin/datasets/search-engine-open-data/}
}
```

For the original search engine project:

```bibtex
@misc{lin2025searchengine,
  author = {Wilson Lin},
  title = {Building a web search engine from scratch in two months with 3 billion neural embeddings},
  year = {2025},
  url = {https://blog.wilsonl.in/search-engine}
}
```

## License

This dataset is made available under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Contact

For questions or issues with this dataset, please open an issue at: https://github.com/wilsonzlin/datasets/search-engine-open-data/