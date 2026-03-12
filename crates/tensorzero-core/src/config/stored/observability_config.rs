use serde::{Deserialize, Serialize};

use crate::config::{BatchWritesConfig, ObservabilityBackend, ObservabilityConfig};

/// Stored version of `ObservabilityConfig`.
///
/// Omits `deny_unknown_fields` so that future fields added in
/// newer versions don't break deserialization in rolled-back gateways.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StoredObservabilityConfig {
    pub enabled: Option<bool>,
    #[serde(default)]
    pub backend: ObservabilityBackend,
    #[serde(default)]
    pub async_writes: bool,
    #[serde(default = "crate::config::default_write_queue_capacity")]
    pub write_queue_capacity: usize,
    #[serde(default)]
    pub batch_writes: BatchWritesConfig,

    /// Deprecated since 2026.2
    #[serde(default)]
    pub disable_automatic_migrations: bool,
}

impl From<ObservabilityConfig> for StoredObservabilityConfig {
    fn from(config: ObservabilityConfig) -> Self {
        let ObservabilityConfig {
            enabled,
            backend,
            async_writes,
            write_queue_capacity,
            batch_writes,
            #[expect(deprecated)]
            disable_automatic_migrations,
        } = config;
        Self {
            enabled,
            backend,
            async_writes,
            write_queue_capacity,
            batch_writes,
            disable_automatic_migrations,
        }
    }
}

impl From<StoredObservabilityConfig> for ObservabilityConfig {
    fn from(stored: StoredObservabilityConfig) -> Self {
        let StoredObservabilityConfig {
            enabled,
            backend,
            async_writes,
            write_queue_capacity,
            batch_writes,
            disable_automatic_migrations,
        } = stored;
        Self {
            enabled,
            backend,
            async_writes,
            write_queue_capacity,
            batch_writes,
            #[expect(deprecated)]
            disable_automatic_migrations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Historical: `disable_automatic_migrations` was an observability field before
    /// being migrated to a top-level `[clickhouse]` section. Stored snapshots from
    /// that era must still parse.
    /// Historical: before `write_queue_capacity` was added, stored configs
    /// didn't include this field. They should still parse with the default value.
    #[test]
    fn test_historical_no_write_queue_capacity() {
        let toml_str = r"
            enabled = true
            async_writes = true
        ";

        let stored: StoredObservabilityConfig =
            toml::from_str(toml_str).expect("should parse without write_queue_capacity");
        let config: ObservabilityConfig = stored.into();
        assert_eq!(
            config.write_queue_capacity,
            crate::config::default_write_queue_capacity(),
            "should use default write_queue_capacity"
        );
    }

    /// Historical: before `flush_concurrency` was added to `BatchWritesConfig`,
    /// stored configs didn't include this field. They should still parse with the default.
    #[test]
    fn test_historical_no_flush_concurrency() {
        let toml_str = r"
            enabled = true

            [batch_writes]
            enabled = true
            flush_interval_ms = 100
            max_rows = 500
        ";

        let stored: StoredObservabilityConfig =
            toml::from_str(toml_str).expect("should parse without flush_concurrency");
        let config: ObservabilityConfig = stored.into();
        assert_eq!(
            config.batch_writes.flush_concurrency,
            crate::config::default_flush_concurrency(),
            "should use default flush_concurrency"
        );
    }

    #[test]
    fn test_disable_automatic_migrations_parses() {
        let toml_str = r"
            enabled = true
            async_writes = true
            disable_automatic_migrations = true
        ";

        let stored: StoredObservabilityConfig =
            toml::from_str(toml_str).expect("should parse deprecated field");
        assert!(
            stored.disable_automatic_migrations,
            "deprecated field should be preserved"
        );
    }
}
