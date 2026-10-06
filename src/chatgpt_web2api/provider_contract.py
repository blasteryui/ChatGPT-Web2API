"""Provider-facing certainty and verification contracts.

These types are intentionally transport-agnostic.  Vision/Core can persist the
receipt returned by the provider without depending on DOM details, while the
legacy Web2API paths remain unchanged unless the strict provider mode is used.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EffectCertainty(StrEnum):
    """Durable certainty about whether a provider mutation happened."""

    KNOWN_NOT_SUBMITTED = "KNOWN_NOT_SUBMITTED"
    DELIVERY_UNCERTAIN = "DELIVERY_UNCERTAIN"
    CONFIRMED_SUBMITTED = "CONFIRMED_SUBMITTED"
    CONFIRMED_COMPLETE = "CONFIRMED_COMPLETE"


class BackendHTTPError(RuntimeError):
    """A non-2xx response from ChatGPT's authenticated backend API."""

    def __init__(
        self,
        endpoint: str,
        status: int,
        body: str = "",
        *,
        retry_after: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.status = int(status)
        self.body = body
        self.retry_after = retry_after
        detail = body[:240].replace("\n", " ").strip()
        suffix = f": {detail}" if detail else ""
        super().__init__(f"Backend HTTP {self.status} for {endpoint}{suffix}")


class ModelSelectionError(RuntimeError):
    """Strict model selection could not positively verify the requested model."""

    def __init__(self, requested_model: str, observed_model: str | None = None) -> None:
        self.requested_model = requested_model
        self.observed_model = observed_model
        super().__init__(
            f"Requested model {requested_model!r} was not positively verified"
            + (f" (observed={observed_model!r})" if observed_model else "")
        )


class ProjectPlacementError(RuntimeError):
    """Strict provider mode landed outside the requested ChatGPT project."""

    def __init__(self, requested_project: str, observed_project: str | None) -> None:
        self.requested_project = requested_project
        self.observed_project = observed_project
        super().__init__(
            f"Requested project {requested_project!r} was not positively verified "
            f"(observed={observed_project!r})"
        )


class RenameVerificationError(RuntimeError):
    """Conversation rename did not survive an independent read-after-write."""

    def __init__(
        self,
        conversation_id: str,
        requested_title: str,
        observed_title: str | None,
    ) -> None:
        self.conversation_id = conversation_id
        self.requested_title = requested_title
        self.observed_title = observed_title
        super().__init__(
            f"Rename verification failed for {conversation_id}: requested "
            f"{requested_title!r}, observed {observed_title!r}"
        )


@dataclass(frozen=True)
class ProviderOperationContext:
    """Caller-owned identity and requested placement/model for one send."""

    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requested_project_id: str | None = None
    requested_model: str | None = None
    observed_project_id: str | None = None
    observed_model: str | None = None


@dataclass
class ProviderOperationReceipt:
    """Structured evidence for one provider send operation.

    The receipt intentionally stores only a hash of the submitted text in the
    turn-anchor summary, not the prompt itself.
    """

    operation_id: str
    conversation_id: str | None
    target_id: str | None
    session_identity: str | None
    requested_project_id: str | None
    observed_project_id: str | None
    requested_model: str | None
    observed_model: str | None
    captured_user_message_id: str | None = None
    turn_anchor: dict[str, Any] | None = None
    response_sha256: str | None = None
    response_status: str = "PREPARED"
    effect_certainty: EffectCertainty = EffectCertainty.KNOWN_NOT_SUBMITTED
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["effect_certainty"] = self.effect_certainty.value
        return data


class ProviderOperationUncertainError(RuntimeError):
    """A send may have reached ChatGPT and must be reconciled, never replayed."""

    def __init__(self, receipt: ProviderOperationReceipt, cause: BaseException) -> None:
        self.receipt = receipt
        self.cause = cause
        super().__init__(
            f"Provider operation {receipt.operation_id} is DELIVERY_UNCERTAIN after "
            f"possible submission: {type(cause).__name__}: {cause}"
        )


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarize_turn_anchor(anchor: Any) -> dict[str, Any]:
    """Serialize correlation identity without copying the raw prompt."""

    return {
        "mode": getattr(anchor, "mode", None),
        "captured_user_message_id": getattr(anchor, "captured_user_message_id", None),
        "latest_user_node_id": getattr(anchor, "latest_user_node_id", None),
        "latest_user_create_time": getattr(anchor, "latest_user_create_time", None),
        "latest_assistant_node_id": getattr(anchor, "latest_assistant_node_id", None),
        "latest_assistant_create_time": getattr(anchor, "latest_assistant_create_time", None),
        "pre_send_wall_time": getattr(anchor, "pre_send_wall_time", None),
        "conversation_id_at_capture": getattr(anchor, "conversation_id_at_capture", None),
        "sent_text_sha256": hash_text(getattr(anchor, "sent_text", "") or ""),
    }
