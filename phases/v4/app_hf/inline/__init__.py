"""Inline V4 + forensic model code bundled with the HF Space.

These modules are copies of phases/v4/src/v4/... and phases/forensic/src/forensic/...
with the registry decorators stripped (we don't need plugin-style architecture in the
demo, the encoders are instantiated directly).

Single source of truth for the model architectures remains the main packages in
phases/v4/src and phases/forensic/src. If those change, mirror the changes here.
"""
