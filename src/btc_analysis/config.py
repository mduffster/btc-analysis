"""Configuration management for BTC analysis pipeline."""

from pathlib import Path
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """Path configuration for data directories."""

    # __file__ is src/btc_analysis/config.py, so we need .parent.parent.parent for project root
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "outputs"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [self.raw_dir, self.processed_dir, self.outputs_dir / "phase1"]:
            dir_path.mkdir(parents=True, exist_ok=True)


class APIConfig(BaseModel):
    """API endpoints and configuration."""

    # IMF SDMX API
    imf_base_url: str = "https://sdmx.imf.org/ws/public/sdmxapi/rest"

    # World Bank API
    world_bank_base_url: str = "https://api.worldbank.org/v2"

    # Chinn-Ito data source
    chinn_ito_url: str = "http://web.pdx.edu/~ito/kaopen_2022.xls"

    # Request timeout
    timeout: int = 60


class AnalysisConfig(BaseModel):
    """Configuration for analysis parameters."""

    # Years to include in analysis
    start_year: int = 2015
    end_year: int = 2023

    # Minimum observations per country
    min_obs: int = 2

    # Significance level for tests
    alpha: float = 0.05


class Config(BaseModel):
    """Main configuration container."""

    paths: PathConfig = Field(default_factory=PathConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


def get_config() -> Config:
    """Get the default configuration."""
    config = Config()
    config.paths.ensure_dirs()
    return config
