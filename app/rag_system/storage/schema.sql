-- =====================================================================
-- RAG System — Snowflake schema
-- =====================================================================
-- Run via `make snowflake-init` (calls rag_system.storage.init_snowflake)
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
-- only loaded when the UI explicitly asks for them.
CREATE TABLE IF NOT EXISTS chunk_images (
    chunk_id    STRING        PRIMARY KEY,
    width       INTEGER,
    height      INTEGER,
    mime_type   STRING        DEFAULT 'image/png',
    image_b64   STRING        -- base64 PNG
);

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
