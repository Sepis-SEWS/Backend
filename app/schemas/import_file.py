from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

FEATURES = (
    "HR",
    "Temp",
    "SBP",
    "DBP",
    "MAP",
    "Resp",
    "O2Sat",
    "WBC",
    "Lactate",
    "Creatinine",
    "Bilirubin_total",
    "Platelets",
    "FiO2",
    "BUN",
    "Glucose",
    "pH",
)
UNITS = {
    "HR": "bpm",
    "Temp": "°C",
    "SBP": "mmHg",
    "DBP": "mmHg",
    "MAP": "mmHg",
    "Resp": "breaths/min",
    "O2Sat": "%",
    "WBC": "10^3/µL",
    "Lactate": "mg/dL",
    "Creatinine": "mg/dL",
    "Bilirubin_total": "mg/dL",
    "Platelets": "10^3/µL",
    "FiO2": "%",
    "BUN": "mg/dL",
    "Glucose": "mg/dL",
    "pH": "dimensionless",
}


class SourceRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    ICULOS: float
    HR: float | None
    Temp: float | None
    SBP: float | None
    DBP: float | None
    MAP: float | None
    Resp: float | None
    O2Sat: float | None
    WBC: float | None
    Lactate: float | None
    Creatinine: float | None
    Bilirubin_total: float | None
    Platelets: float | None
    FiO2: float | None
    BUN: float | None
    Glucose: float | None
    pH: float | None
    # Timestamp parsing is allowed, but numerical coercion is not.
    observed_at: AwareDatetime | None = Field(default=None, strict=False)

    def values(self) -> dict[str, float | None]:
        return self.model_dump(exclude={"observed_at"})


class ReplayProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    source: Literal["physionet2019"]
    dataset_version: str = Field(min_length=1, max_length=32)
    source_set: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    source_record: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    converter: str = Field(min_length=1, max_length=64)
    time_basis: Literal["research_replay"]
    replay_base: AwareDatetime


class ImportFile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    encounter_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    patient_id: str = Field(min_length=1, max_length=128)
    bed: str = Field(min_length=1, max_length=128)
    assignee: str = Field(min_length=1, max_length=128)
    icu_id: Literal["ICU-A"]
    unit_profile: Literal["physionet2019-source-v1"] = "physionet2019-source-v1"
    admission_anchor: AwareDatetime | None = None
    observations: list[SourceRow] = Field(min_length=1, max_length=10000)
    # Optional so existing hand-written files and stored envelopes are unchanged.
    provenance: ReplayProvenance | None = None

    @model_validator(mode="after")
    def chronological_hourly(self) -> "ImportFile":
        explicit = [row.observed_at is not None for row in self.observations]
        if self.provenance is not None and (
            self.admission_anchor is None
            or self.admission_anchor != self.provenance.replay_base
        ):
            raise ValueError(
                "Replay provenance requires admission_anchor == replay_base"
            )
        if self.admission_anchor is not None:
            if any(explicit):
                raise ValueError("Choose admission_anchor OR per-row observed_at")
        elif not all(explicit):
            raise ValueError(
                "Every row needs observed_at when admission_anchor is absent"
            )

        start = self.observations[0].ICULOS
        if start < 1 or start != int(start):
            raise ValueError("Import ICULOS must start at an integer hour >= 1")
        for offset, row in enumerate(self.observations):
            if row.ICULOS != start + offset:
                raise ValueError("Import ICULOS must be a contiguous hourly grid")
        try:
            times = self.times()
        except (OverflowError, ValueError):
            raise ValueError(
                "Import timestamp is outside the supported range"
            ) from None
        for hour, when in enumerate(times, 1):
            if when - times[0] != timedelta(hours=hour - 1):
                raise ValueError("Row timestamps must match elapsed ICULOS hours")
        return self

    def times(self) -> list[datetime]:
        if self.admission_anchor is not None:
            return [
                (self.admission_anchor + timedelta(hours=row.ICULOS)).astimezone(UTC)
                for row in self.observations
            ]
        return [row.observed_at.astimezone(UTC) for row in self.observations]
