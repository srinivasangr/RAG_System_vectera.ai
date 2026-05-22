-- ============================================================================
-- RAG System — full Snowflake schema (idempotent; safe to re-run).
-- Provisions a fresh database: base tables + all columns + vector indexes.
-- ============================================================================

-- =====================================================================

-- 1. Warehouse: small + auto-suspend to save credits
CREATE WAREHOUSE IF NOT EXISTS RAG_WH
    WITH WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- 2. Database + schema
CREATE DATABASE IF NOT EXISTS RAG_DB;
CREATE SCHEMA   IF NOT EXISTS RAG_DB.RAG_SCHEMA;

USE DATABASE RAG_DB;
USE SCHEMA   RAG_SCHEMA;
USE WAREHOUSE RAG_WH;

-- 3. Documents — one row per source PDF
CREATE TABLE IF NOT EXISTS documents (
    doc_id          STRING        PRIMARY KEY,
    source_path     STRING        NOT NULL,
    company         STRING,
    doc_date        DATE,
    doc_type        STRING,        -- 'investor_presentation' | 'third_party_report' | ...
    version_label   STRING,        -- e.g. 'Mar 2026'
    page_count      INTEGER,
    checksum        STRING,        -- sha256 of file; dedupe key
    ingested_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 4. Chunks — text + embedding + metadata
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        STRING        PRIMARY KEY,
    doc_id          STRING        NOT NULL,
    page_number     INTEGER       NOT NULL,
    chunk_index     INTEGER       NOT NULL,
    text            STRING        NOT NULL,
    token_count     INTEGER,
    chunk_type      STRING,        -- 'prose' | 'table' | 'caption'
    embedding       VECTOR(FLOAT, 768),
    -- snapshot company/date here too so we can filter without joining
    company         STRING,
    doc_date        DATE,
    version_label   STRING,
    created_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 5. Chunk images — base64-encoded source images for chart_description chunks
-- Separate table so the chunks table stays narrow and the image bytes are
-- 6. Query log — for audit and eval
CREATE TABLE IF NOT EXISTS query_log (
    query_id        STRING        PRIMARY KEY,
    question        STRING,
    filters         VARIANT,
    retrieved_ids   ARRAY,
    answer          STRING,
    llm_provider    STRING,
    llm_model       STRING,
    latency_ms      INTEGER,
    created_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Snowflake automatically uses an internal vector index for cosine similarity
-- via VECTOR_COSINE_SIMILARITY(embedding, query_vec). No explicit index DDL needed.

-- ---- additional tables & columns ----
-- =====================================================================
-- RAG System — Snowflake schema v2 (additive migration)
-- =====================================================================
-- Run via: python -m rag_system.storage.migrate_v2
--
-- This file is ADDITIVE and IDEMPOTENT. It assumes schema.sql (v1) has
-- already created the warehouse/db/schema + documents/chunks/chunk_images/
-- query_log tables. It then:
--   (a) ALTERs documents/chunks/query_log to add v2 columns
--   (b) CREATEs the new v2 tables (parent_chunks, parent_images,
--       propositions, table_rows, chart_records, ingest_checkpoints)
--
-- All statements use IF NOT EXISTS so re-running is safe.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. documents — add v2 metadata columns
--    (doc_type + version_label already exist from v1)
-- ---------------------------------------------------------------------
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ticker         STRING;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS as_of_date     DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_family_id  STRING;
-- doc_type_conf: classifier confidence 0..1
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type_conf  FLOAT;
-- as_of_source: 'content' | 'filename' | null
ALTER TABLE documents ADD COLUMN IF NOT EXISTS as_of_source   STRING;

-- ---------------------------------------------------------------------
-- 2. chunks — add parent linkage + qualifier/footnote preservation
--    + propagated doc metadata (so retrieval can filter without joins)
-- ---------------------------------------------------------------------
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_id       STRING;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS footnote_text   STRING;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS qualifier_text  STRING;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_type        STRING;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS as_of_date      DATE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_family_id   STRING;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS slide_title     STRING;

-- ---------------------------------------------------------------------
-- 3. query_log — observability columns (per-stage trace, provider chain)
-- ---------------------------------------------------------------------
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS router_intent     STRING;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS sub_queries       VARIANT;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS retrieval_stages  VARIANT;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS rerank_top_ids    VARIANT;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS conflict_pairs    VARIANT;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS provider_chain    VARIANT;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS reasoning_trace   STRING;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS total_latency_ms  INTEGER;

-- ---------------------------------------------------------------------
-- 4. parent_chunks — slide/section-level context for small-to-big
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id      STRING        PRIMARY KEY,
    doc_id         STRING        NOT NULL,
    page_number    INTEGER       NOT NULL,
    slide_title    STRING,
    text           STRING        NOT NULL,    -- full slide/section text (~1500-2500 tok)
    token_count    INTEGER,
    company        STRING,
    doc_type       STRING,
    doc_date       DATE,
    as_of_date     DATE,
    doc_family_id  STRING,
    version_label  STRING,
    created_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------
-- 5. propositions — atomic facts; the primary DENSE retrieval target
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS propositions (
    prop_id        STRING        PRIMARY KEY,
    chunk_id       STRING        NOT NULL,    -- source child chunk
    parent_id      STRING,                    -- for small-to-big expansion
    doc_id         STRING        NOT NULL,
    page_number    INTEGER,
    text           STRING        NOT NULL,    -- single self-contained statement
    embedding      VECTOR(FLOAT, 768),
    -- propagated metadata for fast filtered retrieval (no joins on hot path)
    company        STRING,
    doc_type       STRING,
    doc_date       DATE,
    as_of_date     DATE,
    doc_family_id  STRING,
    version_label  STRING,
    created_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------
-- 6. table_rows — structured table data with column labels preserved
--    Each row keeps every value tied to its column header.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS table_rows (
    row_id         STRING        PRIMARY KEY,
    chunk_id       STRING        NOT NULL,    -- the table chunk this row came from
    doc_id         STRING        NOT NULL,
    page_number    INTEGER,
    table_id       STRING,                    -- groups rows of the same table
    row_idx        INTEGER,
    columns_json   VARIANT,                   -- {"col_label": "value", ...}
    flat_text      STRING,                    -- "Metric: X; PSA: 92.0%; NSA: 84.3%; ..."
    embedding      VECTOR(FLOAT, 768),
    company        STRING,
    doc_type       STRING,
    doc_date       DATE,
    as_of_date     DATE,
    doc_family_id  STRING,
    created_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------
-- 7. chart_records — vision-extracted (label, value) tuples for charts/maps
--    Retrieved by company + label/value text match (no embedding needed).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chart_records (
    record_id      STRING        PRIMARY KEY,
    chunk_id       STRING,                    -- parent chart chunk (nullable)
    doc_id         STRING        NOT NULL,
    page_number    INTEGER,
    chart_id       STRING,                    -- groups records from same subchart
    chart_kind     STRING,                    -- bar | pie | map | logo_table | other
    label          STRING,                    -- e.g. "PSA", "Caesars Palace"
    value          STRING,                    -- e.g. "78%", "39%", "" if name-only
    unit           STRING,                    -- "%", "M sq ft", "GW", ...
    bbox           STRING,                    -- "[x,y,w,h]" — debug only
    confidence     FLOAT,                     -- 0..1; below threshold -> flagged to prompt
    vision_model   STRING,                    -- which model produced it
    company        STRING,
    doc_type       STRING,
    doc_date       DATE,
    as_of_date     DATE,
    doc_family_id  STRING,
    created_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------
-- 8. ingest_checkpoints — per-stage resume-on-crash for ingestion
--    Idempotency key is (doc_id, checksum, stage).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_checkpoints (
    doc_id         STRING        NOT NULL,
    checksum       STRING        NOT NULL,    -- file sha256
    stage          STRING        NOT NULL,    -- parse|chunk|parent|vision|propositions|embed|upsert
    status         STRING        NOT NULL,    -- pending|done|failed
    detail         STRING,                    -- error message / counts / notes
    updated_at     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (doc_id, stage)
);

-- =====================================================================
-- RAG System — Snowflake schema v3 (additive migration)
-- =====================================================================
-- Run via: python -m rag_system.storage.migrate_v2 --file schema_v3.sql
-- Adds: file/PDF metadata on documents, the raw PDF (document_files),
-- per-page images (page_images), and vision-quality columns on chunks.
-- All statements use IF NOT EXISTS so re-running is safe.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. documents — file + PDF metadata; relative stored path (no abs path)
-- ---------------------------------------------------------------------
ALTER TABLE documents ADD COLUMN IF NOT EXISTS original_filename STRING;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS stored_pdf_path   STRING;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_size_bytes   INTEGER;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS mime_type         STRING;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_author        STRING;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_title         STRING;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_created       STRING;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_page_count    INTEGER;

-- ---------------------------------------------------------------------
-- 2. chunks — vision-quality fields
--    chunk_type now includes 'chart' and 'figure' (in addition to prose/table)
-- ---------------------------------------------------------------------
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS confidence  FLOAT;    -- vision conf for visual chunks
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS kind_detail STRING;   -- e.g. bar_chart, world_map, logo_table

-- ---------------------------------------------------------------------
-- 3. document_files — the raw uploaded PDF (provenance / re-processing)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_files (
    doc_id      STRING PRIMARY KEY,
    filename    STRING,
    mime        STRING DEFAULT 'application/pdf',
    size_bytes  INTEGER,
    content_b64 STRING,                       -- base64 of the PDF bytes
    created_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------
-- 4. page_images — one rendered PNG per page (citation modal + re-vision)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS page_images (
    parent_id    STRING PRIMARY KEY,          -- = parent_chunks.parent_id
    doc_id       STRING NOT NULL,
    page_number  INTEGER NOT NULL,
    width        INTEGER,
    height       INTEGER,
    mime_type    STRING DEFAULT 'image/png',
    image_b64    STRING,
    created_at   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ---------------------------------------------------------------------
-- 5. chart_records — add a per-record description (richer than bare value)
-- ---------------------------------------------------------------------
ALTER TABLE chart_records ADD COLUMN IF NOT EXISTS description STRING;
