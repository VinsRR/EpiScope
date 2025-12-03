PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

/* 1) Reviews: one row per source review (aka your paper_id) */
CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY,           -- e.g., "2022_Du_k"
  review_title TEXT NOT NULL,
  review_doi TEXT,
  publication_year INTEGER,
  journal TEXT,
  file_path TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

/* 2) Expansion runs: one row per pipeline execution on a review */
CREATE TABLE IF NOT EXISTS review_expansion_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  review_id TEXT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
  strategy_name TEXT NOT NULL,
  run_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

  -- summary stats from results.summary
  review_type TEXT,
  declared_included_count INTEGER,
  declared_included_evidence TEXT,
  inclusion_criteria_summary TEXT,
  supplements_found INTEGER,
  total_candidates INTEGER,
  candidates_with_matches INTEGER,
  crossref_dois_found INTEGER,
  extraction_payload_json JSON,

  -- pipeline provenance / reproducibility
  pipeline_name TEXT,                              -- optional label
  candidate_generator_class TEXT NOT NULL,         -- e.g. "tableref.candidates.GeminiFullFileGenerator"
  candidate_generator_config JSON NOT NULL,        -- JSON from GeminiConfig (no secrets)
  matcher_class TEXT NOT NULL,                     -- e.g. "tableref.matching.ReferenceMatcher"
  matcher_config JSON NOT NULL,                    -- JSON from SimpleSurnameMatcherConfig
  library_versions JSON,                           -- optional {"tableref":"...", "episcope":"...", "python":"..."}
  prompts_snapshot JSON,                           -- optional {"system_prompt":"...", "user_prompt":"..."}
  config_hash TEXT NOT NULL                        -- deterministic fingerprint of the above
);

CREATE INDEX IF NOT EXISTS idx_exp_runs_review_time
  ON review_expansion_runs (review_id, run_at DESC);
CREATE INDEX IF NOT EXISTS idx_exp_runs_strategy
  ON review_expansion_runs (strategy_name);
CREATE INDEX IF NOT EXISTS idx_exp_runs_config_hash
  ON review_expansion_runs (config_hash);

/* 3) Expansion papers: one row per candidate/matched reference for that run */
CREATE TABLE IF NOT EXISTS review_expansion_papers (
  exp_id INTEGER PRIMARY KEY AUTOINCREMENT,

  -- link to run + source review identity for clarity and filtering
  run_id INTEGER NOT NULL REFERENCES review_expansion_runs(run_id) ON DELETE CASCADE,
  review_id TEXT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,

  -- redundant but *explicit* source review fields for query ergonomics
  review_title TEXT NOT NULL,
  review_doi TEXT,

  -- index/order in the review’s extracted list
  ref_index INTEGER NOT NULL,

  -- raw candidate & matching outcome
  ref_candidate_text TEXT,
  ref_has_match INTEGER NOT NULL CHECK (ref_has_match IN (0,1)),
  ref_match_score REAL,

  -- normalized metadata for the expanded paper (the matched reference)
  ref_title TEXT,
  ref_authors JSON,               -- JSON array of strings
  ref_journal TEXT,
  ref_year INTEGER,
  ref_doi TEXT,                   -- DOI parsed from the review’s reference
  ref_crossref_doi TEXT,          -- DOI resolved via Crossref
  ref_crossref_error TEXT,

  -- prevent duplicates within a run
  UNIQUE (run_id, ref_index)
);

/* 4) Supplements: one row per supplement resource found in a run */
CREATE TABLE IF NOT EXISTS review_expansion_supplements (
    supplement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES review_expansion_runs(run_id) ON DELETE CASCADE,
    s_label TEXT,
    s_href TEXT,
    s_content_note TEXT
);

/* 5) Notes: one row per general note from the extraction */
CREATE TABLE IF NOT EXISTS review_expansion_notes (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES review_expansion_runs(run_id) ON DELETE CASCADE,
    note_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exp_papers_run
  ON review_expansion_papers (run_id);

CREATE INDEX IF NOT EXISTS idx_exp_papers_review
  ON review_expansion_papers (review_id);

CREATE INDEX IF NOT EXISTS idx_exp_papers_refdoi
  ON review_expansion_papers (ref_doi);

CREATE INDEX IF NOT EXISTS idx_exp_papers_crossrefdoi
  ON review_expansion_papers (ref_crossref_doi);

CREATE INDEX IF NOT EXISTS idx_exp_papers_match
  ON review_expansion_papers (ref_has_match, ref_match_score DESC);

CREATE INDEX IF NOT EXISTS idx_exp_supplements_run_id ON review_expansion_supplements(run_id);
CREATE INDEX IF NOT EXISTS idx_exp_notes_run_id ON review_expansion_notes(run_id);
