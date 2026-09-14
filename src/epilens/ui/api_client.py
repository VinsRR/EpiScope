"""HTTP client used by the Streamlit Studio frontend."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class StudioApiError(RuntimeError):
    pass


class StudioApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            timeout=kwargs.pop("timeout", 60),
            **kwargs,
        )
        if not response.ok:
            try:
                payload = response.json()
                detail = payload.get("detail", payload)
            except Exception:
                detail = response.text
            raise StudioApiError(str(detail or f"Request failed ({response.status_code})"))
        if response.status_code == 204:
            return None
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return response.content

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health", timeout=10)

    def workspaces(self) -> Dict[str, Any]:
        return self._request("GET", "/workspaces", timeout=10)

    def create_workspace(self, workspace_id: str, name: str) -> Dict[str, Any]:
        return self._request("POST", "/workspaces", json={"id": workspace_id, "name": name})

    def workspace(self, workspace_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/workspaces/{workspace_id}")

    def update_settings(self, workspace_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        return self._request(
            "PATCH", f"/workspaces/{workspace_id}/settings", json={"changes": changes}
        )

    def documents(self, workspace_id: str) -> list[Dict[str, Any]]:
        return self._request("GET", f"/workspaces/{workspace_id}/documents")["items"]

    def upload_document(
        self, workspace_id: str, *, filename: str, content: bytes, paper_id: str
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/workspaces/{workspace_id}/documents",
            files={"file": (filename, content)},
            data={"paper_id": paper_id},
            timeout=120,
        )

    def papers(self, workspace_id: str) -> list[Dict[str, Any]]:
        return self._request("GET", f"/workspaces/{workspace_id}/papers")["items"]

    def paper(self, workspace_id: str, paper_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/workspaces/{workspace_id}/papers/{paper_id}")

    def index(
        self, workspace_id: str, document_ids: list[str], *, replace_existing: bool
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/workspaces/{workspace_id}/index-jobs",
            json={"document_ids": document_ids, "replace_existing": replace_existing},
        )

    def tasks(self, workspace_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/workspaces/{workspace_id}/tasks")

    def task(self, workspace_id: str, task_key: str) -> Dict[str, Any]:
        return self._request("GET", f"/workspaces/{workspace_id}/tasks/{task_key}")

    def validate_task(self, workspace_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        return self._request(
            "POST", f"/workspaces/{workspace_id}/tasks/validate", json=task
        )

    def save_task(
        self, workspace_id: str, task: Dict[str, Any], *, editing_key: Optional[str] = None
    ) -> Dict[str, Any]:
        if editing_key:
            return self._request(
                "PUT", f"/workspaces/{workspace_id}/tasks/{editing_key}", json=task
            )
        return self._request("POST", f"/workspaces/{workspace_id}/tasks", json=task)

    def explore(self, workspace_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._request(
            "POST", f"/workspaces/{workspace_id}/runs/explore", json=request, timeout=300
        )

    def workflow(
        self,
        workspace_id: str,
        *,
        kind: str,
        paper_ids: list[str],
        task_key: str,
        detailed: bool,
    ) -> Dict[str, Any]:
        endpoint = "classification" if kind == "classification" else "precision-miner"
        return self._request(
            "POST",
            f"/workspaces/{workspace_id}/runs/{endpoint}",
            json={"paper_ids": paper_ids, "task_key": task_key, "detailed": detailed},
        )

    def jobs(self, workspace_id: str) -> list[Dict[str, Any]]:
        return self._request("GET", f"/workspaces/{workspace_id}/jobs")["items"]

    def job(self, workspace_id: str, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/workspaces/{workspace_id}/jobs/{job_id}")

    def cancel_job(self, workspace_id: str, job_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/workspaces/{workspace_id}/jobs/{job_id}/cancel")

    def retry_job(self, workspace_id: str, job_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/workspaces/{workspace_id}/jobs/{job_id}/retry")

    def runs(self, workspace_id: str) -> list[Dict[str, Any]]:
        return self._request("GET", f"/workspaces/{workspace_id}/runs")["items"]

    def run(self, workspace_id: str, run_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/workspaces/{workspace_id}/runs/{run_id}")

    def download_run(self, workspace_id: str, run_id: str, output_format: str) -> bytes:
        response = requests.get(
            f"{self.base_url}/workspaces/{workspace_id}/runs/{run_id}/download",
            params={"format": output_format},
            timeout=60,
        )
        if not response.ok:
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text
            raise StudioApiError(str(detail or "Download failed"))
        return response.content
