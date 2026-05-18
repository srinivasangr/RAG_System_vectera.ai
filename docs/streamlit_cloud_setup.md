# Deploying the hosted version on Streamlit Community Cloud

The hosted version is **read-only**: reviewers can chat against the
pre-ingested corpus, but the upload UI is hidden (Streamlit Cloud's free
tier doesn't have enough memory to run Docling reliably).

## One-time setup (3 minutes)

### 1. Open Streamlit Community Cloud

https://share.streamlit.io → sign in with the same GitHub account that owns
the repo.

### 2. Create a new app

Click **Create app** → **Deploy a public app from GitHub**.

| Field | Value |
|---|---|
| Repository | `srinivasangr/RAG_System_vectera.ai` |
| Branch | `main` |
| Main file path | `app/rag_system/ui/streamlit_app.py` |
| App URL | pick any subdomain (e.g. `rag-investor-docs`) |

Click **Advanced settings**:

| Field | Value |
|---|---|
| Python version | `3.11` |

### 3. Paste secrets

Still in **Advanced settings → Secrets**, paste this TOML block (filling in
your actual values):

```toml
# Snowflake
SNOWFLAKE_ACCOUNT   = "YOUR_ACCOUNT_IDENTIFIER"
SNOWFLAKE_USER      = "YOUR_USERNAME"
SNOWFLAKE_PASSWORD  = "YOUR_PASSWORD"
SNOWFLAKE_ROLE      = "ACCOUNTADMIN"
SNOWFLAKE_WAREHOUSE = "RAG_WH"
SNOWFLAKE_DATABASE  = "RAG_DB"
SNOWFLAKE_SCHEMA    = "RAG_SCHEMA"

# LLM (Cerebras hosts gpt-oss-120b)
CEREBRAS_API_KEY = "csk-..."

# Optional providers
GEMINI_API_KEY   = ""
OPENAI_API_KEY   = ""
ANTHROPIC_API_KEY= ""
OPENROUTER_API_KEY = ""

# Embedding (local sentence-transformers)
EMBEDDING_PROVIDER = "local"
EMBEDDING_MODEL    = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM      = "768"

# LLM default
LLM_PROVIDER = "cerebras"
LLM_MODEL    = "gpt-oss-120b"

# IMPORTANT: hides the upload UI on the hosted version
IS_HOSTED = "true"
```

### 4. Deploy

Click **Deploy**. First boot takes 5-10 minutes (pip-installing PyTorch +
sentence-transformers + Snowflake connector + downloading the BGE-base
model on first query).

### 5. (Snowflake side) Allow Streamlit Cloud IPs

If you've ever set a Snowflake **Network Policy**, you'll need to allowlist
Streamlit Cloud's egress IPs. Default Snowflake accounts have no policy and
accept from anywhere — most users skip this step.

If you do need it, Streamlit's IP ranges are published here:
https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/trust-and-security#ips

In Snowflake:
```sql
CREATE NETWORK POLICY streamlit_cloud
  ALLOWED_IP_LIST = ('<paste Streamlit IPs here>');

ALTER USER YOUR_USERNAME SET NETWORK_POLICY = streamlit_cloud;
```

## After deploy

- **Auto-redeploy on push:** every commit to `main` triggers a redeploy. No
  manual step needed.
- **Logs:** Streamlit Cloud → your app → **Manage app** (top-right) →
  **Logs** for boot logs and runtime errors.
- **Reboot:** **Manage app → Reboot** if things hang.

## Notes / known limits on the hosted version

- **No uploads.** The `is_hosted=true` flag hides the file uploader. Users
  who want to add PDFs run the app locally per the main README.
- **Cold start ~30-60s** the first time after the instance has been idle —
  Streamlit Cloud spins down inactive apps.
- **First query downloads the BGE-base model (~440MB)** to the container's
  local cache. Subsequent queries are instant.
- **Memory budget:** the Community tier gives ~1GB RAM. The query path
  (sentence-transformers + Snowflake + LLM API call) fits comfortably; the
  ingestion path (Docling) does not — which is why uploads are disabled.
