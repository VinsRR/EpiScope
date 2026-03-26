from __future__ import annotations

import json
import os
from typing import Any, Dict

import dotenv
import pandas as pd
import requests
import streamlit as st

dotenv.load_dotenv()


DEFAULT_API_BASE_URL = os.getenv("EPISCOPE_API_BASE_URL", "http://localhost:8000")
print(f"Using API base URL: {DEFAULT_API_BASE_URL}")
print("Mongo URI configured:", bool(os.getenv("MONGO_URI")))
print("Gemini API key configured:", bool(os.getenv("GEMINI_API_KEY")))


def api_request(api_base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{api_base_url.rstrip('/')}{path}",
        json=payload,
        timeout=1800,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        raise RuntimeError(detail or f"Request failed with status {response.status_code}")
    return response.json()


def api_health(api_base_url: str) -> Dict[str, Any]:
    response = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def json_block(value: Any) -> None:
    st.code(json.dumps(value, indent=2, ensure_ascii=False), language="json")


def render_classification_result(result: Dict[str, Any]) -> None:
    if "decision" not in result:
        st.subheader("Classification Result")
        st.json(result)
        return

    decision = result["decision"]
    classification = decision["result"].get("classification", [])
    confidence = decision["result"].get("confidence")
    probabilities = decision["result"].get("class_probabilities", {}) or {}

    st.subheader("Decision")
    st.write(f"Labels: `{', '.join(classification) if classification else 'None'}`")
    st.write(f"Confidence: `{confidence}`")

    if probabilities:
        prob_df = (
            pd.DataFrame(
                [{"label": key, "probability": value} for key, value in probabilities.items()]
            )
            .sort_values("probability", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    metadata = decision.get("metadata") or {}
    if metadata:
        with st.expander("Paper Metadata", expanded=False):
            st.write(f"Title: {metadata.get('title', '')}")
            st.write(f"Abstract: {metadata.get('abstract', '')}")
            if metadata.get("keywords"):
                st.write(f"Keywords: {', '.join(metadata['keywords'])}")

    top_evidence = decision.get("top_evidence") or []
    with st.expander(f"Top Evidence ({len(top_evidence)})", expanded=True):
        for idx, chunk in enumerate(top_evidence, start=1):
            st.markdown(f"**{idx}. {chunk.get('artifacts', {}).get('category', 'unknown')}**")
            st.write(chunk.get("text", ""))
            st.caption(
                f"paper_id={chunk.get('paper_id')} | section={chunk.get('section_type')} | score={chunk.get('rank_score')}"
            )

    provenance = result.get("provenance") or {}
    with st.expander(f"Provenance ({len(provenance.get('evidences', []))} evidences)", expanded=False):
        st.write("Raw answer")
        st.write(provenance.get("answer", ""))
        evidences = provenance.get("evidences", []) or []
        if evidences:
            st.dataframe(pd.DataFrame(evidences), use_container_width=True, hide_index=True)

    trace = result.get("trace") or {}
    with st.expander("Prompt / Response Trace", expanded=False):
        st.write("Raw LLM response")
        st.write(trace.get("raw_llm_response", ""))
        messages = trace.get("prompt_messages") or []
        if messages:
            json_block(messages)

    training = result.get("training") or {}
    with st.expander("Training-Oriented Artifacts", expanded=False):
        st.write(f"Reward: `{training.get('reward')}`")
        st.write(f"Samples collected: `{len(training.get('all_samples', []))}`")
        if training.get("all_samples"):
            json_block(training["all_samples"])


def render_precision_miner_result(result: Dict[str, Any]) -> None:
    if "result" not in result:
        st.subheader("Precision Miner Result")
        st.json(result)
        return

    structured = result["result"]
    st.subheader("Extraction Result")
    st.write(structured.get("description", ""))

    items = structured.get("items") or []
    if items:
        st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
    else:
        st.info("No extraction items were returned.")

    metadata = result.get("metadata") or {}
    if metadata:
        with st.expander("Paper Metadata", expanded=False):
            st.write(f"Title: {metadata.get('title', '')}")
            st.write(f"Abstract: {metadata.get('abstract', '')}")

    with st.expander(f"Relevant Chunks ({len(result.get('relevant_chunks', []))})", expanded=True):
        for idx, chunk in enumerate(result.get("relevant_chunks", []) or [], start=1):
            st.markdown(f"**{idx}. {chunk.get('section_type', 'unknown')}**")
            st.write(chunk.get("text", ""))

    provenance = result.get("provenance") or {}
    with st.expander("Provenance", expanded=False):
        st.write(provenance.get("answer", ""))
        evidences = provenance.get("evidences", []) or []
        if evidences:
            st.dataframe(pd.DataFrame(evidences), use_container_width=True, hide_index=True)

    trace = result.get("trace") or {}
    with st.expander("Prompt / Response Trace", expanded=False):
        st.write(trace.get("raw_llm_response", ""))
        messages = trace.get("prompt_messages") or []
        if messages:
            json_block(messages)


def render_explorer_result(result: Dict[str, Any]) -> None:
    st.subheader("Retrieval")
    st.write(f"Retrieved chunks: `{result.get('retrieval_count', 0)}`")

    retrieved_chunks = result.get("retrieved_chunks") or []
    with st.expander(f"Retrieved Chunks ({len(retrieved_chunks)})", expanded=True):
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**{idx}. {chunk.get('paper_id', 'unknown')}**")
            st.write(chunk.get("text", ""))
            st.caption(
                f"section={chunk.get('section_type')} | source={chunk.get('source')} | score={chunk.get('rank_score')}"
            )

    if result.get("generate_answer"):
        st.subheader("Generated Answer")
        st.write(result.get("answer") or "")

        provenance = result.get("provenance") or {}
        with st.expander("Generation Provenance", expanded=False):
            st.write(provenance.get("answer", ""))
            evidences = provenance.get("evidences", []) or []
            if evidences:
                st.dataframe(pd.DataFrame(evidences), use_container_width=True, hide_index=True)


st.set_page_config(page_title="EpiScope UI", layout="wide")
st.title("EpiScope")
st.caption("Thin Streamlit frontend over the EpiScope API.")

with st.sidebar:
    st.header("Backend")
    api_base_url = st.text_input("API Base URL", value=DEFAULT_API_BASE_URL)

    health = None
    try:
        health = api_health(api_base_url)
        st.success("API reachable")
        checks = health.get("checks", {})
        if not checks.get("mongo_uri_configured"):
            st.warning("Mongo URI is not configured on the API by default. You will likely need to provide it below.")
        if not checks.get("llm_api_key_configured") and checks.get("llm_provider") != "ollama":
            st.warning(f"No API key detected for the default provider `{checks.get('llm_provider')}`.")
        with st.expander("Backend Defaults", expanded=False):
            json_block(health.get("defaults", {}))
        with st.expander("Backend Checks", expanded=False):
            json_block(checks)
    except Exception as exc:
        st.error(f"API not reachable: {exc}")
        st.stop()

    st.header("Runtime Configuration")
    defaults = (health or {}).get("defaults", {})
    default_mongo_uri = defaults.get("mongo_uri")
    strategy_name = st.text_input("Strategy Name", value=defaults.get("strategy_name", "grobid"))
    mongo_uri_placeholder = "******** (loaded from backend env)" if default_mongo_uri else ""
    mongo_uri = st.text_input("Mongo URI", value="", type="password", placeholder=mongo_uri_placeholder)
    mongo_db_name = st.text_input("Mongo DB Name", value=defaults.get("mongo_db_name", "episcope_academic_db"))
    qdrant_url = st.text_input("Qdrant URL", value=defaults.get("qdrant_url", "http://qdrant:6333"))
    qdrant_collection = st.text_input("Qdrant Collection", value=defaults.get("qdrant_collection", "episcope_academic"))
    provider_options = ["gemini", "openai", "openrouter", "ollama"]
    default_provider = defaults.get("llm_provider", "gemini")
    llm_provider = st.selectbox(
        "LLM Provider",
        provider_options,
        index=provider_options.index(default_provider) if default_provider in provider_options else 0,
    )
    llm_model = st.text_input("LLM Model", value=defaults.get("llm_model", "gemini-2.5-flash"))
    llm_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
    workflow_top_k = st.number_input("Workflow Top-K", min_value=1, max_value=100, value=int(defaults.get("workflow_top_k", 10)))
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"],
        index=["dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"].index(defaults.get("retrieval_mode", "hybrid"))
        if defaults.get("retrieval_mode", "hybrid") in ["dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"]
        else 1,
    )
    evidence_reranker_kind = st.selectbox(
        "Evidence Reranker",
        ["none", "global_cross_encoder", "within_label_cross_encoder"],
        index=["none", "global_cross_encoder", "within_label_cross_encoder"].index(defaults.get("evidence_reranker_kind", "none"))
        if defaults.get("evidence_reranker_kind", "none") in ["none", "global_cross_encoder", "within_label_cross_encoder"]
        else 0,
    )
    cross_encoder_model = st.text_input(
        "Cross-Encoder Model",
        value=defaults.get("cross_encoder_model") or os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )
    cross_encoder_top_k = st.number_input("Cross-Encoder Top-K", min_value=1, max_value=100, value=int(defaults.get("cross_encoder_top_k", 15)))

backend_config = {
    "strategy_name": strategy_name,
    "mongo_uri": mongo_uri if mongo_uri != "" else default_mongo_uri,
    "mongo_db_name": mongo_db_name,
    "qdrant_url": qdrant_url,
    "qdrant_collection": qdrant_collection,
    "llm_provider": llm_provider,
    "llm_model": llm_model,
    "llm_temperature": llm_temperature,
    "workflow_top_k": int(workflow_top_k),
    "retrieval_mode": retrieval_mode,
    "evidence_reranker_kind": evidence_reranker_kind,
    "cross_encoder_model": cross_encoder_model or None,
    "cross_encoder_top_k": int(cross_encoder_top_k),
}

if not backend_config["mongo_uri"]:
    st.sidebar.info("Classification and precision miner usually require a Mongo URI so the backend can load paper metadata by paper ID.")

explorer_tab, classification_tab, precision_tab = st.tabs(["Explorer", "Classification", "Precision Miner"])

with explorer_tab:
    st.subheader("Explorer")
    with st.form("explorer-form"):
        query = st.text_area("Query", height=120)
        top_k = st.number_input("Top-K", min_value=1, max_value=100, value=10, key="explorer-top-k")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )
        generate_answer = st.checkbox("Use Generator on Retrieved Chunks", value=False)
        filters_text = st.text_area(
            "Payload Filters (JSON object)",
            value="{}",
            help="Example: {\"year\": 2020, \"section_type\": \"Methods\"}",
            height=120,
        )
        submitted = st.form_submit_button("Run Explorer", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Please provide a query.")
        else:
            try:
                parsed_filters = json.loads(filters_text or "{}")
                if not isinstance(parsed_filters, dict):
                    raise ValueError("Filters must be a JSON object.")
            except Exception as exc:
                st.error(f"Invalid filters JSON: {exc}")
                parsed_filters = None

            if parsed_filters is not None:
                with st.spinner("Running explorer query..."):
                    try:
                        payload = {
                            "query": query.strip(),
                            "top_k": int(top_k),
                            "similarity_threshold": float(similarity_threshold),
                            "generate_answer": generate_answer,
                            "filters": parsed_filters,
                            "config": backend_config,
                        }
                        result = api_request(api_base_url, "/explore", payload)
                        render_explorer_result(result)
                    except Exception as exc:
                        st.error(f"Explorer failed: {exc}")

with classification_tab:
    st.subheader("Paper Classification")
    with st.form("classification-form"):
        paper_id = st.text_input("Paper ID")
        classifier_kind = st.selectbox(
            "Classifier Kind",
            ["data_accessibility", "paper_type", "data_type", "geo"],
            index=0,
        )
        detailed = st.checkbox("Return Detailed Result", value=True)
        submitted = st.form_submit_button("Run Classification", use_container_width=True)

    if submitted:
        if not paper_id.strip():
            st.warning("Please provide a paper ID.")
        else:
            with st.spinner("Running classification..."):
                try:
                    payload = {
                        "paper_id": paper_id.strip(),
                        "classifier_kind": classifier_kind,
                        "detailed": detailed,
                        "config": backend_config,
                    }
                    result = api_request(api_base_url, "/classify", payload)
                    render_classification_result(result)
                except Exception as exc:
                    st.error(f"Classification failed: {exc}")

with precision_tab:
    st.subheader("Precision Miner")
    with st.form("precision-miner-form"):
        paper_id = st.text_input("Paper ID", key="precision-paper-id")
        miner_kind = st.selectbox(
            "Miner Kind",
            ["find_data_sources", "find_supplementary_links", "identify_key_references"],
            index=0,
        )
        detailed = st.checkbox("Return Detailed Result", value=True, key="precision-detailed")
        submitted = st.form_submit_button("Run Precision Miner", use_container_width=True)

    if submitted:
        if not paper_id.strip():
            st.warning("Please provide a paper ID.")
        else:
            with st.spinner("Running precision miner..."):
                try:
                    payload = {
                        "paper_id": paper_id.strip(),
                        "miner_kind": miner_kind,
                        "detailed": detailed,
                        "config": backend_config,
                    }
                    result = api_request(api_base_url, "/precision-miner", payload)
                    render_precision_miner_result(result)
                except Exception as exc:
                    st.error(f"Precision miner failed: {exc}")
