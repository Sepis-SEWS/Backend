"""Strict source-native hourly file contract, separate from serving inputs."""

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

    @model_validator(mode="after")
    def chronological_hourly(self) -> "ImportFile":
        explicit = [row.observed_at is not None for row in self.observations]
        if self.admission_anchor is not None:
            if any(explicit):
                raise ValueError("Choose admission_anchor OR per-row observed_at")
        elif not all(explicit):
            raise ValueError(
                "Every row needs observed_at when admission_anchor is absent"
            )
        for hour, row in enumerate(self.observations, 1):
            if row.ICULOS != hour:
                raise ValueError("Import ICULOS must be the hourly grid 1..n")
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
