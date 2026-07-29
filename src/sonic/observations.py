from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

OBSERVATION_DIMS = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
    "motion_joint_positions_10frame_step1": 290,
    "motion_joint_velocities_10frame_step1": 290,
    "motion_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_lowerbody_10frame_step1": 120,
    "motion_joint_velocities_lowerbody_10frame_step1": 120,
    "smpl_joints_4frame_step1": 288,
    "smpl_anchor_orientation_4frame_step1": 24,
    "motion_joint_positions_wrists_4frame_step1": 24,
}


@dataclass(frozen=True)
class ObservationLayout:
    policy_names: tuple[str, ...]
    encoder_names: tuple[str, ...]
    encoder_dimension: int
    required_g1: frozenset[str]

    @classmethod
    def load(cls, path: Path) -> "ObservationLayout":
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, dict):
            raise ValueError(f"Invalid observation config: {path}")
        encoder = document.get("encoder")
        if not isinstance(encoder, dict):
            raise ValueError("Observation config has no encoder section")

        policy_names = _enabled_names(document.get("observations"), "observations")
        encoder_names = _enabled_names(
            encoder.get("encoder_observations"), "encoder.encoder_observations"
        )
        modes = encoder.get("encoder_modes")
        if not isinstance(modes, list):
            raise ValueError("Observation config has no encoder modes")
        g1_modes = [mode for mode in modes if mode.get("name") == "g1"]
        if len(g1_modes) != 1 or int(g1_modes[0].get("mode_id", -1)) != 0:
            raise ValueError("Expected exactly one G1 encoder mode with mode_id 0")
        required = frozenset(str(name) for name in g1_modes[0]["required_observations"])
        unknown_required = required - set(encoder_names)
        if unknown_required:
            raise ValueError(
                f"G1 mode requires disabled observations: {unknown_required}"
            )

        layout = cls(
            policy_names=policy_names,
            encoder_names=encoder_names,
            encoder_dimension=int(encoder["dimension"]),
            required_g1=required,
        )
        layout._validate_names()
        return layout

    @property
    def policy_input_dimension(self) -> int:
        return sum(OBSERVATION_DIMS[name] for name in self.policy_names)

    @property
    def encoder_input_dimension(self) -> int:
        return sum(OBSERVATION_DIMS[name] for name in self.encoder_names)

    def pack_encoder(self, values: dict[str, np.ndarray]) -> np.ndarray:
        return self._pack(self.encoder_names, values, required=self.required_g1)

    def pack_policy(self, values: dict[str, np.ndarray]) -> np.ndarray:
        return self._pack(self.policy_names, values, required=set(self.policy_names))

    def _pack(
        self,
        names: tuple[str, ...],
        values: dict[str, np.ndarray],
        *,
        required: set[str] | frozenset[str],
    ) -> np.ndarray:
        unexpected = set(values) - set(names)
        if unexpected:
            raise ValueError(
                f"Values supplied for unconfigured observations: {unexpected}"
            )
        missing = required - set(values)
        if missing:
            raise ValueError(f"Missing required observations: {missing}")
        parts: list[np.ndarray] = []
        for name in names:
            value = np.asarray(
                values.get(name, np.zeros(OBSERVATION_DIMS[name])),
                dtype=np.float32,
            ).reshape(-1)
            expected = OBSERVATION_DIMS[name]
            if value.size != expected:
                raise ValueError(f"{name} has {value.size} values; expected {expected}")
            parts.append(value)
        return np.concatenate(parts)[None]

    def _validate_names(self) -> None:
        unknown = (set(self.policy_names) | set(self.encoder_names)) - set(
            OBSERVATION_DIMS
        )
        if unknown:
            raise ValueError(f"Unknown SONIC observations: {unknown}")
        if self.policy_input_dimension != 994:
            raise ValueError(
                f"SONIC decoder layout is {self.policy_input_dimension}, expected 994"
            )
        if self.encoder_dimension != 64:
            raise ValueError(
                f"SONIC token dimension is {self.encoder_dimension}, expected 64"
            )


def _enabled_names(value: Any, section: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Observation config section {section} is not a list")
    return tuple(str(item["name"]) for item in value if item.get("enabled", False))
