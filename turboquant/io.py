from __future__ import annotations

from pathlib import Path

import torch

from turboquant.turboquant_mse import TurboQuantMSE
from turboquant.turboquant_prod import TurboQuantProd
from turboquant.types import IndexArtifact, PackedCodes, QuantizerState, TurboQuantMSEPayload, TurboQuantProdPayload


FORMAT_VERSION = 1


def _serialize_packed_codes(packed: PackedCodes) -> dict[str, object]:
    return {
        "data": packed.data,
        "n_rows": packed.n_rows,
        "dim": packed.dim,
        "bits": packed.bits,
    }


def _deserialize_packed_codes(serialized: dict[str, object]) -> PackedCodes:
    return PackedCodes(
        data=serialized["data"],
        n_rows=int(serialized["n_rows"]),
        dim=int(serialized["dim"]),
        bits=int(serialized["bits"]),
    )


def _serialize_state(state: QuantizerState) -> dict[str, object]:
    return {
        "kind": state.kind,
        "dim": state.dim,
        "bits": state.bits,
        "codebook": state.codebook,
        "rotation": state.rotation,
        "projection": state.projection,
        "scale": state.scale,
        "format_version": state.format_version,
    }


def _deserialize_state(serialized: dict[str, object]) -> QuantizerState:
    return QuantizerState(
        kind=str(serialized["kind"]),
        dim=int(serialized["dim"]),
        bits=int(serialized["bits"]),
        codebook=serialized.get("codebook"),
        rotation=serialized.get("rotation"),
        projection=serialized.get("projection"),
        scale=serialized.get("scale"),
        format_version=int(serialized.get("format_version", FORMAT_VERSION)),
    )


def _serialize_payload(payload: TurboQuantMSEPayload | TurboQuantProdPayload) -> dict[str, object]:
    if isinstance(payload, TurboQuantProdPayload):
        if payload.mse_codes is None or not isinstance(payload.residual_signs, PackedCodes):
            raise ValueError("save_index_artifact expects packed TurboQuantProd payloads")
        return {
            "kind": "turboquant_prod",
            "mse_codes": _serialize_packed_codes(payload.mse_codes),
            "residual_signs": _serialize_packed_codes(payload.residual_signs),
            "residual_norm": payload.residual_norm,
            "n_rows": payload.mse_codes.n_rows,
            "dim": payload.mse_codes.dim,
        }

    if payload.codes is None:
        raise ValueError("save_index_artifact expects packed TurboQuantMSE payloads")
    return {
        "kind": "turboquant_mse",
        "codes": _serialize_packed_codes(payload.codes),
        "n_rows": payload.codes.n_rows,
        "dim": payload.codes.dim,
    }


def _deserialize_payload(serialized: dict[str, object]) -> TurboQuantMSEPayload | TurboQuantProdPayload:
    kind = str(serialized["kind"])
    if kind == "turboquant_prod":
        return TurboQuantProdPayload(
            mse_codes=_deserialize_packed_codes(serialized["mse_codes"]),
            residual_signs=_deserialize_packed_codes(serialized["residual_signs"]),
            residual_norm=serialized["residual_norm"],
        )
    if kind == "turboquant_mse":
        return TurboQuantMSEPayload(codes=_deserialize_packed_codes(serialized["codes"]))
    raise ValueError(f"unsupported payload kind: {kind}")


def _validate_quantizer_payload_match(
    quantizer: TurboQuantMSE | TurboQuantProd,
    payload: TurboQuantMSEPayload | TurboQuantProdPayload,
) -> None:
    if isinstance(quantizer, TurboQuantProd):
        if not isinstance(payload, TurboQuantProdPayload):
            raise ValueError("quantizer and payload kinds must match")
        if payload.mse_codes is None or not isinstance(payload.residual_signs, PackedCodes):
            raise ValueError("save_index_artifact expects packed TurboQuantProd payloads")
        if payload.mse_codes.dim != quantizer.dim or payload.mse_codes.bits != quantizer.bits - 1:
            raise ValueError("quantizer and payload metadata must match")
        if payload.residual_signs.dim != quantizer.dim or payload.residual_signs.bits != 1:
            raise ValueError("quantizer and payload metadata must match")
        if payload.residual_signs.n_rows != payload.mse_codes.n_rows:
            raise ValueError("quantizer and payload row counts must match")
        return

    if not isinstance(payload, TurboQuantMSEPayload):
        raise ValueError("quantizer and payload kinds must match")
    if payload.codes is None:
        raise ValueError("save_index_artifact expects packed TurboQuantMSE payloads")
    if payload.codes.dim != quantizer.dim or payload.codes.bits != quantizer.bits:
        raise ValueError("quantizer and payload metadata must match")


def save_index_artifact(
    path: str | Path,
    *,
    quantizer: TurboQuantMSE | TurboQuantProd,
    payload: TurboQuantMSEPayload | TurboQuantProdPayload,
    metadata: dict[str, object] | None = None,
) -> None:
    _validate_quantizer_payload_match(quantizer, payload)
    state = quantizer.export_state()
    user_metadata = dict(metadata or {})
    normalization = str(user_metadata.get("normalization", "unit_sphere"))
    serialized_payload = _serialize_payload(payload)
    artifact = {
        "format_version": FORMAT_VERSION,
        "state": _serialize_state(state),
        "payload": serialized_payload,
        "metadata": user_metadata,
        "normalization": normalization,
        "algorithm": state.kind,
        "dim": state.dim,
        "bits": state.bits,
        "n_rows": serialized_payload["n_rows"],
    }
    torch.save(artifact, Path(path))


def load_index_artifact(path: str | Path) -> IndexArtifact:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    format_version = int(artifact.get("format_version", FORMAT_VERSION))
    state = _deserialize_state(artifact["state"])
    payload = _deserialize_payload(artifact["payload"])
    metadata = dict(artifact.get("metadata", {}))
    normalization = str(artifact.get("normalization", metadata.get("normalization", "unit_sphere")))

    if state.kind == "turboquant_mse":
        quantizer = TurboQuantMSE.from_state(state)
    elif state.kind == "turboquant_prod":
        quantizer = TurboQuantProd.from_state(state)
    else:
        raise ValueError(f"unsupported quantizer kind: {state.kind}")

    return IndexArtifact(
        state=state,
        payload=payload,
        normalization=normalization,
        metadata=metadata,
        quantizer=quantizer,
        format_version=format_version,
    )
