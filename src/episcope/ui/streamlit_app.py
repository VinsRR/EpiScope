from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import dotenv
import pandas as pd
import streamlit as st

from episcope.ui.api_client import StudioApiClient

dotenv.load_dotenv()

DEFAULT_API_BASE_URL = os.getenv("EPISCOPE_API_BASE_URL", "http://localhost:8000")
TERMINAL_JOB_STATUSES = {
    "cancelled", "completed", "completed_with_errors", "failed", "interrupted"
}

st.set_page_config(page_title="EpiScope Studio", page_icon="🔬", layout="wide")


def json_block(value: Any) -> None:
    st.code(json.dumps(value, indent=2, ensure_ascii=False), language="json")


def get_client() -> StudioApiClient:
    return StudioApiClient(st.session_state.get("api_base_url", DEFAULT_API_BASE_URL))


def selected_workspace_id() -> Optional[str]:
    return st.session_state.get("workspace_id")


def require_workspace() -> str:
    workspace_id = selected_workspace_id()
    if not workspace_id:
        st.info("Create or select a workspace on the Home page first.")
        st.stop()
    return workspace_id


def render_api_error(exc: Exception) -> None:
    st.error(str(exc))


def render_explore_result(result: Dict[str, Any]) -> None:
    st.subheader("Retrieved evidence")
    st.caption(
        f"{result.get('retrieval_count', 0)} chunks"
        + (f" · run {result['run_id'][:8]}" if result.get("run_id") else "")
    )
    if result.get("answer"):
        st.markdown(result["answer"])
    for index, chunk in enumerate(result.get("retrieved_chunks") or [], start=1):
        with st.expander(
            f"{index}. {chunk.get('paper_id', 'unknown')} · "
            f"{chunk.get('section_type', 'Other')}",
            expanded=index <= 3,
        ):
            st.write(chunk.get("text", ""))
            st.caption(
                f"source={chunk.get('source', 'unknown')} · score={chunk.get('rank_score')}"
            )
    if result.get("provenance"):
        with st.expander("Provenance"):
            st.json(result["provenance"])


def render_workflow_result(result: Dict[str, Any]) -> None:
    items = result.get("items", []) if isinstance(result, dict) else []
    errors = result.get("errors", []) if isinstance(result, dict) else []
    for item in items:
        with st.expander(str(item.get("paper_id", "Result")), expanded=True):
            payload = item.get("result", item)
            decision = payload.get("decision") if isinstance(payload, dict) else None
            extraction = payload.get("result") if isinstance(payload, dict) else None
            if decision and isinstance(decision, dict):
                classification = decision.get("result", {}).get("classification", [])
                st.markdown(f"**Labels:** {', '.join(map(str, classification)) or 'None'}")
                st.caption(f"Confidence: {decision.get('result', {}).get('confidence')}")
            elif extraction and isinstance(extraction, dict) and "items" in extraction:
                st.write(extraction.get("description", ""))
                if extraction.get("items"):
                    st.dataframe(pd.DataFrame(extraction["items"]), width="stretch")
            else:
                st.json(payload)
            with st.expander("Complete result"):
                st.json(payload)
    if errors:
        st.error(f"{len(errors)} item(s) failed")
        st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)


def home_page() -> None:
    st.title("EpiScope Studio")
    st.caption("A local workspace for evidence-aware scientific-paper analysis.")
    client = get_client()
    try:
        health = client.health()
        listing = client.workspaces()
    except Exception as exc:
        st.error(f"The EpiScope API is not reachable: {exc}")
        st.code("episcope studio", language="bash")
        return

    checks = health.get("checks", {})
    cols = st.columns(4)
    cols[0].metric("API", "Ready")
    cols[1].metric("LLM provider", checks.get("llm_provider", "unknown"))
    cols[2].metric(
        "LLM access", "Configured" if checks.get("llm_api_key_configured") else "Needs setup"
    )
    cols[3].metric("Workspaces", len(listing.get("items", [])))

    with st.expander("Setup diagnostics"):
        diagnostics = [
            {"Component": "API", "Status": "Ready", "Detail": "Connected"},
            {
                "Component": "Parser",
                "Status": "Ready",
                "Detail": checks.get("parser_backend", "unstructured"),
            },
            {
                "Component": "Vector index",
                "Status": "Ready",
                "Detail": checks.get("vector_backend", "local_file"),
            },
            {
                "Component": "GROBID",
                "Status": "Configured" if checks.get("grobid_url_configured") else "Optional",
                "Detail": "Used only by GROBID workspaces",
            },
            {
                "Component": "Ollama",
                "Status": "Configured" if checks.get("ollama_host_configured") else "Optional",
                "Detail": "Used only with the Ollama provider",
            },
        ]
        st.dataframe(pd.DataFrame(diagnostics), hide_index=True, width="stretch")
        credentials = checks.get("credential_presence") or {}
        if credentials:
            configured = [name for name, present in credentials.items() if present]
            st.caption(
                "Credential presence: "
                + (", ".join(configured) if configured else "no hosted-provider keys detected")
            )

    if not checks.get("llm_api_key_configured"):
        st.warning(
            "Generated answers and workflows need the selected provider's environment "
            "variable, or a running Ollama server. Secrets are never entered in Studio."
        )

    st.subheader("Create a workspace")
    with st.form("create-workspace"):
        workspace_id = st.text_input(
            "Workspace ID", placeholder="systematic-review", help="Used in URLs and on disk."
        )
        workspace_name = st.text_input("Display name", placeholder="Systematic Review")
        submitted = st.form_submit_button("Create workspace", type="primary")
    if submitted:
        try:
            created = client.create_workspace(workspace_id.strip(), workspace_name.strip())
            st.session_state.workspace_id = created["id"]
            st.success(f"Created {created['name']}")
            st.rerun()
        except Exception as exc:
            render_api_error(exc)

    if selected_workspace_id():
        try:
            workspace = client.workspace(selected_workspace_id() or "")
            st.subheader(workspace["name"])
            st.caption(workspace["path"])
            metrics = st.columns(3)
            metrics[0].metric("Documents", workspace["document_count"])
            metrics[1].metric("Papers", workspace["paper_count"])
            metrics[2].metric("Custom tasks", workspace["task_count"])
        except Exception as exc:
            render_api_error(exc)


def library_page() -> None:
    st.title("Library")
    workspace_id = require_workspace()
    client = get_client()

    st.subheader("Upload a document")
    upload = st.file_uploader(
        "PDF, text, or Markdown", type=["pdf", "txt", "md", "text"], accept_multiple_files=False
    )
    if upload is not None:
        proposed_id = st.text_input(
            "Paper ID",
            value=Path(upload.name).stem,
            key=f"paper-id-{getattr(upload, 'file_id', upload.name)}",
        )
        if st.button("Upload", type="primary", width="stretch"):
            try:
                result = client.upload_document(
                    workspace_id,
                    filename=upload.name,
                    content=upload.getvalue(),
                    paper_id=proposed_id.strip(),
                )
                if result.get("duplicate"):
                    st.info("That exact file is already in this workspace.")
                else:
                    st.success("Document uploaded. Select it below to index it.")
                st.rerun()
            except Exception as exc:
                render_api_error(exc)

    try:
        documents = client.documents(workspace_id)
    except Exception as exc:
        render_api_error(exc)
        return

    st.subheader("Documents")
    if not documents:
        st.info("Upload your first paper to begin.")
    else:
        display = pd.DataFrame(documents)[
            ["id", "original_name", "paper_id", "status", "title", "section_count", "updated_at"]
        ]
        st.dataframe(display, width="stretch", hide_index=True)
        labels = {
            item["id"]: f"{item['original_name']} · {item['paper_id']} · {item['status']}"
            for item in documents
        }
        selected = st.multiselect(
            "Documents to index", options=list(labels), format_func=lambda key: labels[key]
        )
        replace_existing = st.checkbox(
            "Replace papers that are already indexed",
            help="Replacement is staged and the previous searchable index is restored if it fails.",
        )
        if st.button("Start indexing", disabled=not selected, type="primary"):
            try:
                job = client.index(workspace_id, selected, replace_existing=replace_existing)
                st.session_state.last_job_id = job["id"]
                st.success(f"Index job {job['id'][:8]} queued.")
            except Exception as exc:
                render_api_error(exc)

    st.subheader("Indexed papers")
    try:
        papers = client.papers(workspace_id)
    except Exception as exc:
        render_api_error(exc)
        papers = []
    if papers:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "sections": paper["section_count"],
                        "references": paper["reference_count"],
                        "source": paper["source"],
                    }
                    for paper in papers
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        paper_id = st.selectbox("Inspect paper", [paper["paper_id"] for paper in papers])
        if paper_id:
            try:
                detail = client.paper(workspace_id, paper_id)
                with st.expander("Metadata", expanded=True):
                    st.json(detail["metadata"])
                with st.expander(f"Sections ({len(detail['sections'])})"):
                    for section in detail["sections"]:
                        st.markdown(f"**{section.get('title') or section.get('section_type', 'Section')}**")
                        st.write(section.get("content", ""))
            except Exception as exc:
                render_api_error(exc)
    else:
        st.info("No indexed papers yet.")


def explore_page() -> None:
    st.title("Explore")
    workspace_id = require_workspace()
    client = get_client()
    try:
        papers = client.papers(workspace_id)
    except Exception as exc:
        render_api_error(exc)
        return
    if not papers:
        st.info("Index at least one paper in the Library first.")
        return

    with st.form("explore"):
        query = st.text_area("Question or search query", height=110)
        cols = st.columns(3)
        top_k = cols[0].number_input("Top-K", 1, 100, 10)
        threshold = cols[1].slider("Similarity threshold", 0.0, 1.0, 0.0, 0.05)
        generate = cols[2].checkbox("Generate answer", value=True)
        paper_filter = st.multiselect(
            "Restrict to papers",
            [paper["paper_id"] for paper in papers],
            help="The local payload filter currently supports one paper at a time.",
        )
        section_filter = st.text_input("Section type", placeholder="Methods")
        submitted = st.form_submit_button("Run", type="primary", width="stretch")
    if submitted:
        if not query.strip():
            st.warning("Enter a question or query.")
        elif len(paper_filter) > 1:
            st.warning("Select at most one paper for the current local filter.")
        else:
            filters: Dict[str, Any] = {}
            if paper_filter:
                filters["paper_id"] = paper_filter[0]
            if section_filter.strip():
                filters["section_type"] = section_filter.strip()
            try:
                with st.spinner("Searching the indexed evidence..."):
                    st.session_state.explore_result = client.explore(
                        workspace_id,
                        {
                            "query": query.strip(),
                            "top_k": int(top_k),
                            "similarity_threshold": float(threshold),
                            "generate_answer": generate,
                            "filters": filters,
                        },
                    )
            except Exception as exc:
                render_api_error(exc)
    if st.session_state.get("explore_result"):
        render_explore_result(st.session_state.explore_result)


def workflows_page() -> None:
    st.title("Workflows")
    workspace_id = require_workspace()
    client = get_client()
    try:
        papers = client.papers(workspace_id)
        tasks = client.tasks(workspace_id)
    except Exception as exc:
        render_api_error(exc)
        return
    if not papers:
        st.info("Index at least one paper in the Library first.")
        return

    with st.form("workflow"):
        mode = st.radio("Workflow", ["Classification", "Precision miner"], horizontal=True)
        family = "classifiers" if mode == "Classification" else "miners"
        specs = tasks[family]
        labels = {item["key"]: item["label"] for item in specs}
        task_key = st.selectbox("Task", list(labels), format_func=lambda key: labels[key])
        selected = st.multiselect(
            "Papers", [paper["paper_id"] for paper in papers], default=[papers[0]["paper_id"]]
        )
        detailed = st.checkbox("Keep detailed evidence and trace", value=True)
        submitted = st.form_submit_button("Queue workflow", type="primary", width="stretch")
    if submitted:
        try:
            job = client.workflow(
                workspace_id,
                kind="classification" if mode == "Classification" else "precision_miner",
                paper_ids=selected,
                task_key=task_key,
                detailed=detailed,
            )
            st.session_state.last_job_id = job["id"]
            st.success(f"Job {job['id'][:8]} queued. Follow it in Jobs & Results.")
        except Exception as exc:
            render_api_error(exc)


def _blank_task(kind: str = "classifier") -> Dict[str, Any]:
    if kind == "classifier":
        return {
            "key": "my_classifier",
            "kind": "classifier",
            "label": "My classifier",
            "description": "",
            "top_k": 10,
            "labels": [
                {"code": "yes", "name": "Yes", "definition": "", "examples": []},
                {"code": "unclear", "name": "Unclear", "definition": "", "examples": []},
            ],
            "multi_label": False,
            "default_label": "unclear",
            "retrieval_templates": [],
            "section_filters": None,
            "system_prompt": None,
            "user_prompt_template": None,
        }
    return {
        "key": "my_miner",
        "kind": "miner",
        "label": "My miner",
        "description": "",
        "top_k": 15,
        "labels": [],
        "multi_label": True,
        "default_label": None,
        "retrieval_templates": ["What evidence answers this extraction task?"],
        "section_filters": None,
        "system_prompt": None,
        "user_prompt_template": None,
    }


def task_builder_page() -> None:
    st.title("Task Builder")
    workspace_id = require_workspace()
    client = get_client()
    try:
        catalog = client.tasks(workspace_id)
    except Exception as exc:
        render_api_error(exc)
        return
    custom = [
        item for family in ("classifiers", "miners") for item in catalog[family]
        if item.get("source") == "workspace"
    ]
    cols = st.columns([3, 1, 1, 1])
    selected = cols[0].selectbox(
        "Workspace task", [""] + [item["key"] for item in custom],
        format_func=lambda key: "Select a task…" if not key else key,
    )
    if cols[1].button("Load", disabled=not selected):
        try:
            st.session_state.task_draft = client.task(workspace_id, selected)
            st.session_state.task_editing_key = selected
            st.rerun()
        except Exception as exc:
            render_api_error(exc)
    if cols[2].button("New"):
        st.session_state.task_draft = _blank_task()
        st.session_state.task_editing_key = None
        st.rerun()
    if cols[3].button("Duplicate", disabled=not selected):
        try:
            duplicated = client.task(workspace_id, selected)
            duplicated["key"] = f"{duplicated['key']}_copy"
            duplicated["label"] = f"{duplicated['label']} (copy)"
            st.session_state.task_draft = duplicated
            st.session_state.task_editing_key = None
            st.rerun()
        except Exception as exc:
            render_api_error(exc)

    draft = st.session_state.get("task_draft") or _blank_task()
    advanced = st.toggle("Edit raw JSON", value=False)
    if advanced:
        raw = st.text_area(
            "Task JSON", value=json.dumps(draft, indent=2), height=500, key="task-raw-json"
        )
        try:
            candidate = json.loads(raw)
            parse_error = None
        except Exception as exc:
            candidate = draft
            parse_error = str(exc)
        if parse_error:
            st.error(parse_error)
    else:
        kind = st.radio(
            "Kind", ["classifier", "miner"],
            index=0 if draft.get("kind") == "classifier" else 1,
            horizontal=True,
        )
        key = st.text_input("Key", value=draft.get("key", ""))
        label = st.text_input("Label", value=draft.get("label", ""))
        description = st.text_area("Description", value=draft.get("description", ""))
        top_k = st.number_input("Top-K", 1, 100, int(draft.get("top_k") or 10))
        candidate = {**draft, "kind": kind, "key": key, "label": label,
                     "description": description, "top_k": int(top_k)}
        if kind == "classifier":
            label_rows = []
            for item in draft.get("labels") or []:
                label_rows.append(
                    {
                        "code": item.get("code", ""),
                        "name": item.get("name", ""),
                        "definition": item.get("definition", ""),
                        "examples": "\n".join(item.get("examples") or []),
                    }
                )
            edited = st.data_editor(
                pd.DataFrame(label_rows), num_rows="dynamic", width="stretch",
                column_config={"examples": st.column_config.TextColumn("Examples (one per line)")},
            )
            candidate["labels"] = [
                {
                    "code": str(row.get("code", "")).strip(),
                    "name": str(row.get("name", "")).strip(),
                    "definition": str(row.get("definition", "")).strip(),
                    "examples": [line.strip() for line in str(row.get("examples", "")).splitlines() if line.strip()],
                }
                for row in edited.to_dict("records") if str(row.get("code", "")).strip()
            ]
            candidate["multi_label"] = st.checkbox(
                "Allow multiple labels", value=bool(draft.get("multi_label", True))
            )
            codes = [item["code"] for item in candidate["labels"]]
            current_default = draft.get("default_label")
            candidate["default_label"] = st.selectbox(
                "Default label", [None] + codes,
                index=([None] + codes).index(current_default) if current_default in codes else 0,
            )
        else:
            templates = st.text_area(
                "Retrieval questions (one per line)",
                value="\n".join(draft.get("retrieval_templates") or []),
                height=140,
            )
            candidate["retrieval_templates"] = [
                line.strip() for line in templates.splitlines() if line.strip()
            ]
            filters = st.text_input(
                "Section filters (comma-separated)",
                value=", ".join(draft.get("section_filters") or []),
            )
            candidate["section_filters"] = [
                item.strip() for item in filters.split(",") if item.strip()
            ] or None
            candidate["labels"] = []
            candidate["default_label"] = None

        with st.expander("Prompt overrides"):
            candidate["system_prompt"] = st.text_area(
                "System prompt", value=draft.get("system_prompt") or ""
            ) or None
            candidate["user_prompt_template"] = st.text_area(
                "User prompt template", value=draft.get("user_prompt_template") or "", height=160
            ) or None

    actions = st.columns(3)
    if actions[0].button("Validate", width="stretch"):
        try:
            validated = client.validate_task(workspace_id, candidate)
            st.session_state.task_draft = validated["task"]
            st.success("Task definition is valid.")
        except Exception as exc:
            render_api_error(exc)
    if actions[1].button("Save", type="primary", width="stretch"):
        try:
            saved = client.save_task(
                workspace_id, candidate, editing_key=st.session_state.get("task_editing_key")
            )
            st.session_state.task_draft = saved
            st.session_state.task_editing_key = saved["key"]
            st.success(f"Saved {saved['key']}")
        except Exception as exc:
            render_api_error(exc)
    actions[2].download_button(
        "Download JSON",
        data=json.dumps(candidate, indent=2),
        file_name=f"{candidate.get('key') or 'task'}.json",
        mime="application/json",
        width="stretch",
    )


def jobs_results_page() -> None:
    st.title("Jobs & Results")
    workspace_id = require_workspace()
    client = get_client()

    @st.fragment(run_every=2)
    def live_jobs() -> None:
        try:
            jobs = client.jobs(workspace_id)
        except Exception as exc:
            render_api_error(exc)
            return
        if not jobs:
            st.info("No jobs yet.")
            return
        for job in jobs[:20]:
            with st.container(border=True):
                cols = st.columns([2, 2, 4, 1])
                cols[0].markdown(f"**{job['kind']}** · `{job['id'][:8]}`")
                cols[1].write(job["status"])
                cols[2].write(job.get("message") or "")
                total = max(1, int(job.get("total") or 1))
                st.progress(min(1.0, int(job.get("current") or 0) / total))
                if job["status"] in {"queued", "running"}:
                    if cols[3].button("Cancel", key=f"cancel-{job['id']}"):
                        client.cancel_job(workspace_id, job["id"])
                        st.rerun(scope="fragment")
                elif job["status"] in {"failed", "interrupted", "cancelled", "completed_with_errors"}:
                    if cols[3].button("Retry", key=f"retry-{job['id']}"):
                        client.retry_job(workspace_id, job["id"])
                        st.rerun(scope="fragment")
                if job.get("error"):
                    st.error(job["error"])

    live_jobs()

    st.subheader("Run history")
    try:
        runs = client.runs(workspace_id)
    except Exception as exc:
        render_api_error(exc)
        return
    if not runs:
        st.info("Completed searches and workflows will appear here.")
        return
    labels = {
        run["id"]: f"{run['created_at'][:19]} · {run['kind']} · {run['status']} · {run['summary']}"
        for run in runs
    }
    run_id = st.selectbox("Result", list(labels), format_func=lambda key: labels[key])
    if run_id:
        try:
            run = client.run(workspace_id, run_id)
            if run["kind"] == "explore":
                render_explore_result(run["result"])
            else:
                render_workflow_result(run["result"])
            cols = st.columns(2)
            cols[0].download_button(
                "Download JSON",
                data=json.dumps(run["result"], indent=2, ensure_ascii=False),
                file_name=f"{run_id}.json",
                mime="application/json",
                width="stretch",
            )
            try:
                csv_data = client.download_run(workspace_id, run_id, "csv")
                cols[1].download_button(
                    "Download CSV", data=csv_data, file_name=f"{run_id}.csv",
                    mime="text/csv", width="stretch",
                )
            except Exception as exc:
                cols[1].caption(f"CSV export unavailable: {exc}")
        except Exception as exc:
            render_api_error(exc)


def settings_page() -> None:
    st.title("Settings")
    workspace_id = require_workspace()
    client = get_client()
    try:
        workspace = client.workspace(workspace_id)
    except Exception as exc:
        render_api_error(exc)
        return
    settings = workspace["settings"]
    st.caption("API keys and connection credentials remain in the server environment.")
    with st.form("runtime-settings"):
        provider_options = ["gemini", "openai", "openrouter", "anthropic", "ollama"]
        provider = st.selectbox(
            "LLM provider", provider_options,
            index=provider_options.index(settings["llm_provider"]),
        )
        model = st.text_input("LLM model", value=settings["llm_model"])
        temperature = st.slider(
            "Temperature", 0.0, 2.0, float(settings["llm_temperature"]), 0.05
        )
        top_k = st.number_input("Workflow Top-K", 1, 100, int(settings["workflow_top_k"]))
        reranker_options = ["none", "global_cross_encoder", "within_label_cross_encoder"]
        reranker = st.selectbox(
            "Evidence reranker",
            reranker_options,
            index=reranker_options.index(settings["evidence_reranker_kind"]),
        )
        cross_encoder_model = st.text_input(
            "Cross-encoder model", value=settings.get("cross_encoder_model") or ""
        )
        cross_encoder_top_k = st.number_input(
            "Cross-encoder Top-K", 1, 100, int(settings["cross_encoder_top_k"])
        )
        submitted = st.form_submit_button("Save runtime settings", type="primary")
    if submitted:
        try:
            client.update_settings(
                workspace_id,
                {
                    "llm_provider": provider,
                    "llm_model": model.strip(),
                    "llm_temperature": temperature,
                    "workflow_top_k": int(top_k),
                    "evidence_reranker_kind": reranker,
                    "cross_encoder_model": cross_encoder_model.strip() or None,
                    "cross_encoder_top_k": int(cross_encoder_top_k),
                },
            )
            st.success("Settings saved.")
            st.rerun()
        except Exception as exc:
            render_api_error(exc)

    st.subheader("Index configuration")
    if workspace["indexed"]:
        st.info("Indexing settings are locked because this workspace already has an index.")
    index_settings = {
        key: settings[key]
        for key in (
            "loader", "chunker", "min_chunk_size", "chunk_size", "chunk_overlap",
            "embed_model", "embed_provider", "index_backend", "retrieval_mode",
        )
    }
    st.json(index_settings)
    if not workspace["indexed"]:
        with st.form("index-settings"):
            loader_options = ["unstructured", "grobid"]
            loader = st.selectbox(
                "Parser", loader_options, index=loader_options.index(settings["loader"])
            )
            chunker_options = ["none", "sentence", "paragraph", "fixed_size"]
            chunker = st.selectbox(
                "Chunker", chunker_options, index=chunker_options.index(settings["chunker"])
            )
            min_chunk_size = st.number_input(
                "Minimum paragraph size", 1, 10000, int(settings["min_chunk_size"])
            )
            chunk_size = st.number_input("Fixed chunk size", 1, 100000, int(settings["chunk_size"]))
            chunk_overlap = st.number_input(
                "Fixed chunk overlap", 0, 99999, int(settings["chunk_overlap"])
            )
            embed_model = st.text_input("Embedding model", value=settings["embed_model"])
            embed_provider = st.selectbox(
                "Embedding provider",
                ["auto", "fastembed", "huggingface", "openai", "gemini", "ollama"],
                index=["auto", "fastembed", "huggingface", "openai", "gemini", "ollama"].index(
                    settings["embed_provider"]
                ),
            )
            save_index_settings = st.form_submit_button("Save index settings")
        if save_index_settings:
            try:
                client.update_settings(
                    workspace_id,
                    {
                        "loader": loader,
                        "chunker": chunker,
                        "min_chunk_size": int(min_chunk_size),
                        "chunk_size": int(chunk_size),
                        "chunk_overlap": int(chunk_overlap),
                        "embed_model": embed_model.strip(),
                        "embed_provider": embed_provider,
                    },
                )
                st.success("Index settings saved.")
                st.rerun()
            except Exception as exc:
                render_api_error(exc)


# Common application frame ---------------------------------------------------
PAGE_FUNCTIONS = {
    "home": home_page,
    "library": library_page,
    "explore": explore_page,
    "workflows": workflows_page,
    "task_builder": task_builder_page,
    "jobs_results": jobs_results_page,
    "settings": settings_page,
}


def run_app() -> None:
    with st.sidebar:
        st.header("EpiScope")
        api_base_url = st.text_input(
            "API URL", value=st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
        )
        st.session_state.api_base_url = api_base_url
        try:
            workspace_listing = get_client().workspaces()
            workspace_items = workspace_listing.get("items", [])
            workspace_labels = {item["id"]: item["name"] for item in workspace_items}
            options = list(workspace_labels)
            current = st.session_state.get("workspace_id")
            if current not in options:
                current = workspace_listing.get("active_workspace_id")
            if current not in options and options:
                current = options[0]
            if options:
                chosen = st.selectbox(
                    "Workspace", options,
                    index=options.index(current) if current in options else 0,
                    format_func=lambda key: workspace_labels[key],
                )
                st.session_state.workspace_id = chosen
            else:
                st.caption("No workspaces yet")
                st.session_state.workspace_id = None
        except Exception:
            st.caption("API unavailable")

    pages = {
        "Studio": [
            st.Page(home_page, title="Home", icon=":material/home:", default=True),
            st.Page(library_page, title="Library", icon=":material/library_books:"),
            st.Page(explore_page, title="Explore", icon=":material/search:"),
            st.Page(workflows_page, title="Workflows", icon=":material/account_tree:"),
        ],
        "Manage": [
            st.Page(task_builder_page, title="Task Builder", icon=":material/edit_note:"),
            st.Page(jobs_results_page, title="Jobs & Results", icon=":material/history:"),
            st.Page(settings_page, title="Settings", icon=":material/settings:"),
        ],
    }
    st.navigation(pages).run()


test_page = os.getenv("EPISCOPE_STUDIO_TEST_PAGE")
if test_page:
    PAGE_FUNCTIONS[test_page]()
else:
    run_app()
