"""C2PA (Coalition for Content Provenance and Authenticity) metadata handling.

C2PA metadata is embedded in PNG files as a JUMBF container chunk
(``caBX``).  This module can detect, extract, and re-inject those
chunks.  Supported issuers:

- Google Imagen
- Adobe Firefly
- Microsoft Designer
- OpenAI (ChatGPT, GPT-4o, Sora, DALL-E)
- Truepic (signing authority)

The parser uses byte-level scanning — it does not validate JUMBF/CBOR
structure but reliably identifies known signatures, issuers, tools,
and actions.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

from remove_ai_watermarks.noai.constants import (
    C2PA_ACTIONS,
    C2PA_AI_TOOLS,
    C2PA_CHUNK_TYPE,
    C2PA_ISSUERS,
    C2PA_SIGNATURES,
    C2PA_SOFT_BINDINGS,
    PNG_SIGNATURE,
    SYNTHID_C2PA_ISSUERS,
)


def has_c2pa_metadata(image_path: Path) -> bool:
    """
    Check if an image contains C2PA metadata.

    Args:
        image_path: Path to the image file.

    Returns:
        True if C2PA metadata is detected, False otherwise.
    """
    image_path = Path(image_path)

    if image_path.suffix.lower() != ".png":
        return False

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return False

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(length)
                    # Check for any C2PA signature
                    for sig in C2PA_SIGNATURES:
                        if sig in chunk_data:
                            return True
                    # Also check if chunk_data itself contains C2PA-like patterns
                    if b"jumb" in chunk_data.lower() or b"c2pa" in chunk_data.lower():
                        return True
                    f.read(4)
                else:
                    f.read(length + 4)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return False


def extract_c2pa_info(image_path: Path) -> dict[str, Any]:
    """
    Extract basic C2PA metadata information from an image.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary containing C2PA metadata info.
    """
    c2pa_info: dict[str, Any] = {}

    if not has_c2pa_metadata(image_path):
        return c2pa_info

    c2pa_info["has_c2pa"] = True
    c2pa_info["type"] = "C2PA (Coalition for Content Provenance and Authenticity)"

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return c2pa_info

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(length)
                    _parse_c2pa_chunk(chunk_data, c2pa_info)
                    f.read(4)
                else:
                    f.read(length + 4)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return c2pa_info


def cbor_text_after(payload: bytes, key: bytes) -> str | None:
    """Return the CBOR text-string immediately following ``key`` in ``payload``.

    Handles CBOR major-type 3 length prefixes: direct (0x60-0x77), 1-byte
    (0x78 NN), and 2-byte (0x79 NN NN). This reads the actual encoded value, so
    it avoids the byte-grabbing artifacts a loose regex produces (e.g. the
    leading length byte showing up as ``fGPT-4o``).
    """
    idx = payload.find(key)
    if idx < 0:
        return None
    p = idx + len(key)
    if p >= len(payload):
        return None
    head = payload[p]
    if 0x60 <= head <= 0x77:
        length, start = head - 0x60, p + 1
    elif head == 0x78 and p + 1 < len(payload):
        length, start = payload[p + 1], p + 2
    elif head == 0x79 and p + 2 < len(payload):
        length, start = (payload[p + 1] << 8) | payload[p + 2], p + 3
    else:
        return None
    raw_str = payload[start : start + length]
    try:
        return raw_str.decode("utf-8")
    except UnicodeDecodeError:
        return raw_str.decode("latin1", errors="replace")


def synthid_verdict(vendors: str) -> str:
    """Human-readable SynthID-source verdict, shared by all callers."""
    return f"likely present ({vendors} embeds SynthID with C2PA)"


def synthid_vendors_in(buffer: bytes) -> list[str]:
    """Return SynthID-using C2PA issuer names whose signature appears in ``buffer``.

    Shared by the PNG caBX parser and the format-agnostic binary scan so both
    apply the same SYNTHID_C2PA_ISSUERS rule against their respective bytes.
    """
    return sorted({name for sig, name in C2PA_ISSUERS.items() if sig in buffer and sig in SYNTHID_C2PA_ISSUERS})


def soft_binding_vendors_in(buffer: bytes) -> list[str]:
    """Return forensic-watermark vendor names whose C2PA soft-binding ``alg``
    identifier appears in ``buffer``.

    A ``c2pa.soft-binding`` assertion names the watermark scheme that stamped the
    pixels (Adobe TrustMark, Digimarc, Imatag, Steg.AI, ...). Shared by the PNG
    caBX parser and the format-agnostic binary scan so both apply the same
    C2PA_SOFT_BINDINGS rule against their respective bytes.
    """
    return sorted({name for sig, name in C2PA_SOFT_BINDINGS.items() if sig in buffer})


def _parse_c2pa_chunk(chunk_data: bytes, c2pa_info: dict[str, Any]) -> None:
    """Parse C2PA chunk data and populate info dictionary."""
    c2pa_info["c2pa_manifest"] = f"C2PA manifest ({len(chunk_data)} bytes)"

    # Find issuers
    issuers: list[str] = []
    for sig, name in C2PA_ISSUERS.items():
        if sig in chunk_data:
            issuers.append(name)
    if issuers:
        c2pa_info["issuer"] = ", ".join(set(issuers))

    # Find AI tools
    ai_tools: list[str] = []
    for sig, name in C2PA_AI_TOOLS.items():
        if sig in chunk_data:
            ai_tools.append(name)
    if ai_tools:
        c2pa_info["ai_tool"] = ", ".join(set(ai_tools))

    # Claim generator and spec version: read the CBOR text-string values
    # directly (regex byte-grabbing produced artifacts like ``fGPT-4o``).
    # Guard with isprintable(): on some manifests (e.g. Microsoft Designer) the
    # first ``name`` key precedes a binary field (a hash), not the generator
    # string, which would otherwise surface as control-char garbage.
    if (generator := cbor_text_after(chunk_data, b"name")) and generator.isprintable():
        c2pa_info["claim_generator"] = generator
    if (spec := cbor_text_after(chunk_data, b"specVersion")) and spec.isprintable():
        c2pa_info["c2pa_spec"] = spec

    # Find actions
    actions: list[str] = []
    for sig, name in C2PA_ACTIONS.items():
        if sig in chunk_data:
            actions.append(name)
    if actions:
        c2pa_info["actions"] = ", ".join(actions)

    # Find timestamps
    timestamp_matches = re.findall(rb"(\d{14}Z)", chunk_data)
    if timestamp_matches:
        c2pa_info["timestamp"] = timestamp_matches[0].decode("utf-8")
        if len(timestamp_matches) > 1:
            c2pa_info["timestamps"] = [t.decode("utf-8") for t in timestamp_matches[:3]]

    # Find digital source type
    ai_source = False
    if b"trainedAlgorithmicMedia" in chunk_data:
        c2pa_info["source_type"] = "trainedAlgorithmicMedia (AI-generated)"
        ai_source = True
    elif b"algorithmicMedia" in chunk_data:
        c2pa_info["source_type"] = "algorithmicMedia"
    elif b"compositeWithTrainedAlgorithmicMedia" in chunk_data:
        c2pa_info["source_type"] = "compositeWithTrainedAlgorithmicMedia (AI-enhanced)"
        ai_source = True

    # SynthID pixel-watermark proxy: a C2PA manifest from a SynthID-using
    # vendor (Google/OpenAI) on AI-generated content implies an invisible
    # SynthID watermark in the pixels (see SYNTHID_C2PA_ISSUERS).
    synthid_vendors = synthid_vendors_in(chunk_data)
    if synthid_vendors and ai_source:
        c2pa_info["synthid_vendors"] = synthid_vendors
        c2pa_info["synthid_watermark"] = synthid_verdict(", ".join(synthid_vendors))

    # Soft-binding: a forensic/third-party watermark vendor named in the
    # manifest (Adobe TrustMark, Digimarc, ...), independent of the issuer.
    soft_binding_vendors = soft_binding_vendors_in(chunk_data)
    if soft_binding_vendors:
        c2pa_info["soft_binding_vendors"] = soft_binding_vendors
        c2pa_info["soft_binding"] = ", ".join(soft_binding_vendors)


def extract_c2pa_chunk(image_path: Path) -> bytes | None:
    """
    Extract the raw C2PA JUMBF chunk from a PNG file.

    Args:
        image_path: Path to the source PNG file.

    Returns:
        Raw bytes of the C2PA chunk or None.
    """
    if image_path.suffix.lower() != ".png":
        return None

    try:
        with open(image_path, "rb") as f:
            signature = f.read(8)
            if signature != PNG_SIGNATURE:
                return None

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break

                length = struct.unpack(">I", chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]

                if chunk_type == C2PA_CHUNK_TYPE:
                    chunk_data = f.read(length)
                    crc = f.read(4)

                    # Check for any C2PA signature
                    for sig in C2PA_SIGNATURES:
                        if sig in chunk_data:
                            return chunk_header + chunk_data + crc

                    # Also check lowercase variants
                    if b"jumb" in chunk_data.lower() or b"c2pa" in chunk_data.lower():
                        return chunk_header + chunk_data + crc
                else:
                    f.read(length + 4)

                if chunk_type == b"IEND":
                    break
    except Exception:
        pass

    return None


def inject_c2pa_chunk(target_path: Path, output_path: Path, c2pa_chunk: bytes) -> None:
    """
    Inject a C2PA JUMBF chunk into a PNG file.

    Args:
        target_path: Path to the target PNG file.
        output_path: Path where the output file will be saved.
        c2pa_chunk: Raw bytes of the C2PA chunk to inject.

    Raises:
        ValueError: If not PNG files.
    """
    if target_path.suffix.lower() != ".png" or output_path.suffix.lower() != ".png":
        raise ValueError("C2PA chunk injection is only supported for PNG files")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "rb") as f_in, open(output_path, "wb") as f_out:
        f_out.write(f_in.read(8))

        c2pa_injected = False
        while True:
            chunk_header = f_in.read(8)
            if len(chunk_header) < 8:
                break

            length = struct.unpack(">I", chunk_header[:4])[0]
            chunk_type = chunk_header[4:8]
            chunk_data = f_in.read(length)
            crc = f_in.read(4)

            if chunk_type == b"IDAT" and not c2pa_injected:
                f_out.write(c2pa_chunk)
                c2pa_injected = True

            if chunk_type == C2PA_CHUNK_TYPE:
                continue

            f_out.write(chunk_header)
            f_out.write(chunk_data)
            f_out.write(crc)

            if chunk_type == b"IEND":
                break
