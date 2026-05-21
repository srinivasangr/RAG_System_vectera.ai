-- =====================================================================
-- RAG System — Snowflake schema v3 (additive migration)
-- =====================================================================
-- Run via: python -m rag_system.storage.migrate_v2 --file schema_v3.sql
-- Adds: file/PDF metadata on documents, the raw PDF (document_files),
-- per-page images (page_images), and vision-quality columns on chunks.
-- All statements use IF NOT EXISTS so re-running is safe.
-- =====================================================================

USE DATABASE RAG_DB;
USE SCHEMA   RAG_SCHEMA;
USE WAREHOUSE RAG_WH;

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
